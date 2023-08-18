# -*- coding: utf-8 -*-

'''
This python script trains N_train networks using Adam method on lbd*|f| + |g| for each lbd in L_lambda
This is done for several functions c1,c2,c3
Then it plots the error on c according to lambda
The goal is to show that there is no clear way to chose lbd a priori
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from lib.nn_siren import SirenNet
from lib.network_tools import new_net, new_optim
from lib.tools import gradient,finite_difference_laplacian_1d

# Functions

def c1(x):
	return torch.cos(2*np.pi*x) + 0.5*torch.cos(4*np.pi*x)

def c2(x):
	return torch.exp(-torch.abs(x))

def u1(x):
	return torch.cos(2*np.pi*x)

def u2(x):
	return torch.sin(2*np.pi*x)

def ddu1(x):
	return -4 * np.pi**2 * torch.cos(2*np.pi*x) 

def ddu2(x):
	return -4 * np.pi**2 * torch.sin(2*np.pi*x)

# Code

use_finite_differences = False

def eval_loss():
	res = net(x_p)
	u = res[:,0].flatten()
	ddu = 0
	if use_finite_differences:
		ddu = finite_difference_laplacian_1d(net,x_p,2*delta).flatten()
	else:
		ddu = gradient(gradient(u,x_p),x_p).flatten()
	c = res[:,1].flatten()
	phi = c*u - ddu
	loss = lbd * torch.nn.functional.mse_loss(net(x_m)[:,0].flatten(),U) + torch.nn.functional.mse_loss(phi,PHI)
	optim.zero_grad()
	loss.backward()
	return loss

def train_net(epochs):
	global net
	global optim
	net = new_net(dim_out=2,layers=3,neurons=16)
	optim = new_optim(net)
	for i in range(epochs):
		optim.step(eval_loss)
		if eval_loss() < 1e-5:
			break
	return net

def net_err_c(net,c):
	x = torch.linspace(0,1,10000).unsqueeze(dim=1)
	return torch.nn.functional.mse_loss(c(x).flatten(),net(x)[:,1].flatten())

# Listing all fucntions
Lc = [[u1,ddu1,c1],[u2,ddu2,c2]]

# Sampling points
N_m = 100
delta = 1e-2
x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)

N_p = 250
x_p = torch.linspace(0,1,N_p,requires_grad=True).unsqueeze(dim=1)

epochs = 2500
N_train = 5
L_lambda = np.logspace(-4,4,30)
lbd = 0

# Init global variables for network and optimizer
net = 0
optim = 0

C_err_plot = np.zeros((len(Lc),len(L_lambda),N_train))
C_err_lbd  = np.zeros((len(Lc),len(L_lambda)))

print("Number of trainings :",len(Lc)*len(L_lambda)*N_train)
for k in range(len(Lc)):
	f = Lc[k]
	u_real = f[0]
	ddu_real = f[1]
	c = f[2]

	U = u_real(x_m).flatten().detach() + torch.randn(N_m)*delta
	PHI = (-ddu_real(x_p) + c(x_p)*u_real(x_p)).flatten().detach()
	for i in range(len(L_lambda)):
		lbd = L_lambda[i]
		serr = 0
		for j in range(N_train):
			net = train_net(epochs)
			serr += float(net_err_c(net,c))
			C_err_plot[k,i,j] = float(net_err_c(net,c))
		C_err_lbd[k,i] = lbd
		print('Functions', str(u_real.__name__) + ' and ' + str(c.__name__), '- lambda = {:.4e}'.format(lbd), '- error on c = {:.4e}'.format(np.mean(C_err_plot[k,i])))

# PLOTS
fig,ax = plt.subplot_mosaic("A;B")

ax["A"].plot(C_err_lbd[0],np.mean(C_err_plot[0],axis=1),'bo')
ax["A"].errorbar(C_err_lbd[0],np.mean(C_err_plot[0],axis=1),fmt='b',lw=1,yerr=[np.min(C_err_plot[0],axis=1),np.max(C_err_plot[0],axis=1)])
ax["A"].set_xscale('log')
ax["A"].set_xlabel(r'$\lambda$')
ax["A"].set_ylabel('Error on c')
ax["A"].set_title(r'c = $\cos(2\pi x) + \frac{1}{2}\cos(4\pi x)$' + '\n' + r'$u = \cos(2\pi x)$')

ax["B"].plot(C_err_lbd[1],np.mean(C_err_plot[1],axis=1),'ro')
ax["B"].errorbar(C_err_lbd[1],np.mean(C_err_plot[1],axis=1),fmt='r',lw=1,yerr=[np.min(C_err_plot[1],axis=1),np.max(C_err_plot[1],axis=1)])
ax["B"].set_xscale('log')
ax["B"].set_xlabel(r'$\lambda$')
ax["B"].set_ylabel('Error on c')
ax["B"].set_title(r'c = $\exp(-|x|)$' + '\n' + r'$u = \sin(2\pi x)$')

title = 'C error for different u and c '
if use_finite_differences:
	title += 'using finite differences'
else:
	title += 'using auto-differentiation'

fig.suptitle(title)
plt.show()