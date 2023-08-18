# -*- coding: utf-8 -*-

'''
This python script implements the DFL method and plots the results of it
'''

from lib.problem import *
from lib.tools import gradient
from lib.network_tools import new_net, update_single_param, num_param
from lib.nn_siren import SirenNet

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

def expansion_step(net,a,p,index,gamma,eps):
	delta = 0.2
	p_loss = seq_penalty(net,eps)
	update_single_param(net,index,a*p)
	new_p_loss = seq_penalty(net,eps)
	update_single_param(net,index,-a*p)
	while new_p_loss <= p_loss - gamma*(a**2):
		a /= delta
		# This should never happen
		if(a > 1e18):
			print('\033[31m'+"Too large a in expansion step, should see what to do"+'\033[0m')
			print(p_loss)
			return a
		update_single_param(net,index,a*p)
		new_p_loss = seq_penalty(net,eps)
		update_single_param(net,index,-a*p)
	return a*delta

def DFL_optim(net,p,a,eps,gamma,theta,nu_f,epochs):
	'''
	net   : the model to train
	p     : exponent for epsilon in step 2
	a     : initial line search length (array of length N)
	eps   : the initial epsilon for step 2
	gamma : parameter decide if we should do an expansion step
	theta : parameter to update epsilon and alpha
	nu_f  : function of one parameter 'k' that returns nu_k
	'''
	l_gerr,l_ferr,l_eps,l_a = [],[],[],[]
	best_f = 1e10
	N = num_param(net)
	d = np.ones(N)

	for k in range(epochs):
		# For display purposes
		best_f = min(f_error(net),best_f)
		l_gerr.append(g_penalty(net))
		l_ferr.append(f_error(net))
		l_eps.append(1/eps)
		l_a.append(np.max(a))
		print("Step",k,"|f| =",f_error(net),"|g| =",g_penalty(net)," P(net) =",seq_penalty(net,eps),"| range a = [",np.min(a),",",np.max(a),"] | epsilon =",eps)
		
		if(np.max(a) < 1e-3):
			return l_gerr,l_ferr,l_eps,l_a

		for i in range(N):

			# Step 1 : minimization on the cone
			p_ini = seq_penalty(net,eps)
			update_single_param(net,i,a[i]*d[i])
			p_plus = seq_penalty(net,eps)
			# 1.2
			if p_plus <= p_ini - gamma*(a[i]**2):
				update_single_param(net,i,-a[i]*d[i])
				a_tmp = expansion_step(net,a[i],d[i],i,gamma,eps)
				update_single_param(net,i,a_tmp*d[i])
				a[i] = a_tmp
				if seq_penalty(net,eps) > p_ini:
					print("	Wrong step in 1.2, substep",i,"diff :", float(seq_penalty(net,eps) - p_ini))
			else:
				update_single_param(net,i,-2*a[i]*d[i])
				p_minus = seq_penalty(net,eps)
				# 1.3
				if p_minus <= p_ini - gamma*(a[i]**2):
					update_single_param(net,i,a[i]*d[i])
					a_tmp = expansion_step(net,a[i],-d[i],i,gamma,eps)
					a[i] = a_tmp
					d[i] = -d[i]
					update_single_param(net,i,a[i]*d[i])
					if(seq_penalty(net,eps) > p_ini):
						print("	Wrong step in 1.3, substep",i,"diff :", seq_penalty(net,eps) - p_ini)
				else:
					# 1.4
					update_single_param(net,i,a[i]*d[i])
					a[i] = theta*a[i]
		# Step 2
		if np.max(a) < eps**p and max(g_penalty(net),0) > nu_f(k):
			eps *= theta
	return l_gerr,l_ferr,l_eps,l_a

def f_error(net):
	return float(torch.nn.functional.mse_loss(net(x_m)[:,0],U))

def g_penalty(net):
	delta = 1e-3
	l = net(x_p)
	u = l[:,0].flatten()
	c = l[:,1].flatten()
	ddu = gradient(gradient(u,x_p),x_p).flatten()
	return float(torch.nn.functional.mse_loss(-ddu + c*u, PHI - delta))

def seq_penalty(net,eps):
	q = 1.1
	return f_error(net) + 1/eps *max(g_penalty(net),0)**q

# Sampling points
N_m = 50
delta = 1e-3
x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)
U = u_real(x_m) + torch.randn(N_m).unsqueeze(dim=1)*delta
U = U.flatten()

# PDE points
N_p = 1000
x_p = torch.linspace(0,1,N_p,requires_grad=True).unsqueeze(dim=1)
PHI = phi_real(x_p).flatten()

net = new_net(dim_out=2,layers=3,neurons=16)

# Parameters
def nu_f(k):
	return (0.9)**k

a0 = 1e-2
alpha0 = np.full(num_param(net),a0)
p=2
eps=1
gamma=1e-6
theta=0.5

l_gerr,l_ferr,l_eps,l_a = DFL_optim(net,p=p,a=alpha0,eps=eps,gamma=gamma,theta=theta,nu_f=nu_f,epochs=10000)
l_epochs = range(len(l_ferr))

xe = torch.linspace(0,1,10000).unsqueeze(dim=1)
print('C error :', torch.nn.functional.mse_loss(net(xe)[:,1].flatten(),c_real(xe).flatten()))

# Plots
N = 1000
x = torch.linspace(0,1,N,requires_grad=True).unsqueeze(dim=1)
U_plot = u_real(x).flatten().detach().numpy()
U_net = net(x)[:,0].flatten().detach().numpy()

C_plot = c_real(x).flatten().detach().numpy()
C_net = net(x)[:,1].flatten().detach().numpy()
x = x.flatten().detach().numpy()

fig,ax = plt.subplot_mosaic("AAC;AAD;BBE;BBF")

ax["A"].plot(x,U_plot,lw=3,label='True u')
ax["A"].plot(x,U_net,lw=2,ls='dashed',label='Approximated u')

ax["B"].plot(x,C_plot,lw=3,label='True c')
ax["B"].plot(x,C_net,lw=2,ls='dashed',label='Approximated c')

ax["C"].plot(l_epochs,l_ferr,label="f")
ax["C"].set_yscale('log')
ax["D"].plot(l_epochs,l_gerr,label="g")
ax["D"].set_yscale('log')
ax["E"].plot(l_epochs,l_eps,label=r'$\frac{1}{\epsilon}$')
ax["E"].set_yscale('log')
ax["F"].plot(l_epochs,l_a,label='max(a)')
ax["F"].set_yscale('log')

for a in ax:
	ax[a].legend()
plt.suptitle("DFL method\n" + r'$\alpha_0 = $' + str(a0) + r'; $\epsilon_0 =$' + str(eps) + r'; $p =$ ' + str(p) + r'; $\gamma =$' + str(gamma) + r'; $\theta=$ ' +str(theta))
plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.3, wspace=0.2)
plt.show()