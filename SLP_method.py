# -*- coding: utf-8 -*-
# This code is a modified version of code made by Mattéo Clémot
# http://perso.ens-lyon.fr/matteo.clemot/

'''
This python script implements the SLP method and plots the results of it
'''


from lib.problem import *
from lib.tools import gradient,finite_difference_laplacian_1d
from lib.network_tools import new_net, update_param, nn_norm
from lib.nn_siren import SirenNet

from scipy.optimize import linprog

import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_KKT(net,lobj,gobj,lcon,gcon,res,prnt=False):
    eps = 1e-3
    f1 = float(torch.max(torch.abs(gobj + res.ineqlin.marginals[0]*gcon)))
    f2 = float(torch.abs(lcon*res.ineqlin.marginals[0]))
    if prnt:
        print("Checking optimality conditions :")
        print(f1,f2)
        print("Checking 3.3.1 : ", max(f1,f2), "?<", eps*float(1+nn_norm(net)))
        print("Checking 3.3.2 : ", float(lcon), "?<", eps*float(1+nn_norm(net)))
    return (max(f1,f2) < eps*(1+float(np.linalg.norm(res.ineqlin.marginals)))) and (float(lcon) < eps*float(1+nn_norm(net)))

def SLP_ineq_optim(obj, con, epsilon, net:SirenNet, epochs=10000, rhobad=1/8, rhogood=3/4, maxStep=100):
    """
    obj : objective function (f(theta), typically the error on samples plus the error on boundary)
    con : condition function in the slp (g(theta), typically the PDE error)
    epsilon : the tolerance for the PDE error
    net : the network
    epochs : the number of epochs
    rhobad, rhogood : the tolerances for rho (ratio between effective increase and predicted increase)
    maxStep : number of consecutive rejected steps after which the algorithm should stop
    """
    l_ferr,l_gerr,l_perr = [],[],[]

    delta = 1 		# Tolerance for the step size
    nu = 1 			# PDE error penalization

    # Main loop
    for k in range(epochs):
        lobj,gobj = g_grad(net, obj)
        lcon,gcon = g_grad(net, con)
        l = lobj + max(0, lcon-epsilon)
        n = gobj.shape[0]
        l_ferr.append(float(lobj))
        l_gerr.append(float(lcon))

        with torch.no_grad():
            prnt = False # Unused
            i = 0
            # Main loop
            while True:
                i += 1
                if i==maxStep:
                    # print('\033[31m'+"Rejected a hundred consecutive steps"+'\033[0m')
                    check_KKT(net,lobj,gobj,lobj,gobj,res,prnt=False)
                    l_perr.append(float(lobj)+nu*float(lcon))
                    return l_ferr,l_gerr,l_perr

                # We linearize the problem
                c = np.concatenate((gobj.cpu().numpy(), [nu]))
                bounds = [(-delta,delta) for _ in range(n)]
                bounds.append((0,None))
                A_ub = np.concatenate((gcon.cpu().numpy(), [-1])).reshape((1,-1))
                b_ub = epsilon-float(lcon)
                # Solve the linear program
                res = linprog(c, bounds=bounds, A_ub=A_ub, b_ub=b_ub,method='highs-ds')
                d = torch.tensor(res.x[:-1], device=device)
                # Update the network acoordingly
                update_param(net, d)
                
                # compute the true error
                with torch.set_grad_enabled(True):
                    nl = obj(net) + max(0,con(net)-epsilon)
                
                # Compute ratio between the true decrease and the expected decrease 
                rho = (l-nl)/(nu*max(0,lcon-epsilon)-res.fun)
     
                # If the step is rejected, we reduce the trust region 
                if rho < rhobad:
                    delta /= 1.2
                    update_param(net, -d)
                    continue
                # If the step is accepted we increase the trust region
                elif rho > rhogood:
                    delta *= 1.2

                l_perr.append(float(lobj)+nu*float(lcon))
                nu_ = max(np.abs(res.ineqlin.marginals[0]), np.max(np.abs(res.lower.marginals[-1])))
                if nu_ > nu:
                    nu = nu_
                break
            for p in net.parameters():
                p.grad.zero_()
    check_KKT(net,lobj,gobj,lcon,gcon,res,prnt=False)
    return l_ferr,l_gerr,l_perr

# Computes the gradient of f with respect to the parameters of net
def g_grad(net, f):
    l = f(net)
    l.backward(retain_graph=True)
    res = torch.cat([p.grad.detach().flatten() for p in net.parameters()])
    for p in net.parameters():
        p.grad.zero_()
    return l,res

# Objective function : error on the samples
def obj_fun(net:SirenNet):
	return torch.nn.functional.mse_loss(net(x_m)[:,0],U.flatten())

# Constraints function : PDE remainder
def con_fun(net:SirenNet):
    l = net(x_p)
    u = l[:,0].reshape([-1,1])
    c = l[:,1].reshape([-1,1])
    # ddu = gradient(gradient(u,x_p),x_p)
    ddu = finite_difference_laplacian_1d(net,x_p,float(obj_fun(net))).unsqueeze(dim=1)
    phi_pred = c*u - ddu
    return torch.nn.functional.mse_loss(PHI,phi_pred)


# Sampling points
N_m = 100
delta = 0
x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)
U = u_real(x_m) + torch.randn(N_m).unsqueeze(dim=1)*delta

# PDE points
N_p = 1000
x_p = torch.linspace(0,1,N_p,requires_grad=True).unsqueeze(dim=1)
PHI = phi_real(x_p)

# The network
net = new_net(dim_out=2,neurons=16,layers=3)

# Parameters
epsilon = 1e-2
F,G,P = SLP_ineq_optim(obj_fun,con_fun,epsilon,net,epochs=10000)

# Computing errors
x_e = torch.linspace(0,1,1000).unsqueeze(dim=1)
print("U loss   :",float(obj_fun(net)))
print("PDE loss :",float(con_fun(net)))
print("C loss   :",float(torch.nn.functional.mse_loss(net(x_e)[:,1].flatten(),c_real(x_e).flatten())))

E = np.arange(len(F))

plotting = False
if plotting:
    # Plots
    N = 10000
    x = torch.linspace(0,1,N).unsqueeze(dim=1)
    U_plot = u_real(x).flatten().detach().numpy()
    U_net  = net(x)[:,0].flatten().detach().numpy()

    C_plot = c_real(x).flatten().detach().numpy()
    C_net = net(x)[:,1].flatten().detach().numpy()
    x = x.flatten().detach().numpy()

    fig,ax = plt.subplot_mosaic("AAC;AAC;AAD;BBD;BBE;BBE")

    ax["A"].plot(x,U_plot,lw=3,label='True u')
    ax["A"].plot(x,U_net,lw=2,ls='dashed',label='Approximated u')

    ax["B"].plot(x,C_plot,lw=3,label='True c')
    ax["B"].plot(x,C_net,lw=2,ls='dashed',label='Approximated c')

    ax["C"].plot(E,F,label='f')
    ax["C"].set_yscale('log')

    ax["D"].plot(E,G,label='g')
    ax["D"].set_yscale('log')

    ax["E"].plot(E,P,label=r'$p=f+\nu g$')
    ax["E"].set_yscale('log')

    for a in ax:
    	ax[a].legend()
    plt.suptitle("Sequential linear programming \n" + r'$\epsilon = $' + str(epsilon))
    plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.3, wspace=0.2)
    plt.show()