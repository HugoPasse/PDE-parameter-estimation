# -*- coding: utf-8 -*-

'''
This is an implementation of the final method for monovariate functions
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

from lib.problem import *
from lib.nn_siren import SirenNet
from lib.fourier_tools import derivative_from_fft_1d
from lib.network_tools import new_net, new_optim, num_param, lp_norm

import sys

plotting = 1

N_m = 20		# Number of samples on U
delta = 5e-2	# Noise on U

if len(sys.argv) >= 3:
	N_m = int(sys.argv[1])
	delta = float(sys.argv[2])

if len(sys.argv) >= 4:
	plotting = int(sys.argv[3])

print('N :', N_m, 'delta :',delta)

x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)
U = u_real(x_m) + torch.randn(N_m).unsqueeze(dim=1)*delta
U = U.detach()

def eval_loss_u():
	loss = torch.nn.functional.mse_loss(net_u(x_m),U)
	optim_u.zero_grad()
	loss.backward()
	return loss

N_p = 10000 # Number of points in which we evaluate the PDE
x_p = torch.linspace(0,1,N_p,requires_grad=True).unsqueeze(dim=1)

net_u = new_net(dim_out=1,neurons=16,layers=3)
optim_u = new_optim(net_u)

# Training U
epochs_u = 5000
llossu = []
for i in range(epochs_u):
	optim_u.step(eval_loss_u)
	llossu.append(eval_loss_u())
	if llossu[-1] < max(delta**2,1e-5):
		break

print('u loss :',float(eval_loss_u()))

p = 100
u_norm = lp_norm(net_u,p,n=1000000)
idealxi = 2*np.pi*(u_norm / (2*delta))**(1/p)
maxerr = 2*(u_norm**(2/p))*((2*delta)**(1-2/p))

if plotting == 1:
	print('U Lp norm                  :',u_norm)
	print('Theorical xi max           :',idealxi)
	print('Theorical max error on DDU :',maxerr)

# Computing the Laplacian with truncated Fourier
N = 1000000
mxi = idealxi
x_fft = torch.linspace(0,1,N).unsqueeze(dim=1)

u_fft = net_u(x_fft).flatten().detach().numpy()
ddu_fft = derivative_from_fft_1d(u_fft,mxi,k=2)
print('MSE on DDU       :',np.mean((ddu_real(x_fft).flatten().detach().numpy() - ddu_fft)**2))
print('Max error on DDU :',np.max((ddu_real(x_fft).flatten().detach().numpy() - ddu_fft)**2))

# Finding the trusted intervals
u_tol = 0.5
l = np.abs(net_u(x_p).flatten().detach().numpy())
l = np.where(l > u_tol)

# Creating the data to train C
x_p = torch.tensor(x_p[l].detach().numpy(),requires_grad=True)
PHI = phi_real(x_p).detach()

ddu_net = torch.tensor(ddu_fft[[i*N//N_p for i in range(N_p)]][l]).unsqueeze(dim=1).detach()
u_net = net_u(x_p).detach()

def eval_loss_c():
	phi = net_c(x_p)*u_net - ddu_net
	loss = torch.nn.functional.mse_loss(phi.float(),PHI)
	optim_c.zero_grad()
	loss.backward()
	return loss

net_c = new_net(dim_out=1,neurons=16,layers=3)
optim_c = new_optim(net_c)

# Training C
epochs_c = 10000
llossc = []
for i in range(epochs_c):
	optim_c.step(eval_loss_c)
	llossc.append(eval_loss_c())
print('PDE loss :',float(eval_loss_c()))


xe = torch.linspace(0,1,10000).unsqueeze(dim=1)
print('c error :',float(torch.nn.functional.mse_loss(c_real(xe),net_c(xe))))


if plotting == 1:
	# Plotting
	N_plot = 10000
	x_plot = torch.linspace(0,1,N_plot).unsqueeze(dim=1)
	C = c_real(x_plot).flatten().detach().numpy()
	C_net = net_c(x_plot).flatten().detach().numpy()
	C_noisy = (phi_real(x_plot).flatten().detach().numpy() + ddu_fft[[i*N//N_plot for i in range(N_plot)]])/net_u(x_plot).flatten().detach().numpy()

	U_net = net_u(x_plot).flatten().detach().numpy()
	U_plot = u_real(x_plot).flatten().detach().numpy()
	DDU =  ddu_real(x_fft).flatten().detach().numpy()

	x_plot = x_plot.detach()

	fig,ax = plt.subplot_mosaic("AAB;AAC")

	ax["A"].plot(x_plot,C,label='True c',lw=5)
	ax["A"].plot(x_plot,C_net,ls='dashed',label='Network c',lw=5)
	ax["A"].plot(x_plot,C_noisy,ls='dotted',label='Noisy solution',lw=4)
	ax["A"].scatter(x_p.flatten().detach().numpy(),[np.min(C)-0.7]*len(x_p),label='x_p',lw=5)
	ax["A"].set_ylim(np.min(C)-1,np.max(C)+1)
	ax["A"].legend()

	ax["B"].plot(x_plot,U_plot,label='True u',lw=3)
	ax["B"].plot(x_plot,U_plot,ls='dashed',label='Network u',lw=3)
	ax["B"].scatter(x_m.flatten().detach().numpy(),U.flatten().detach().numpy(),color='lightgreen',marker='+',label='Samples',zorder=3)
	ax["B"].scatter(x_p.flatten().detach().numpy(),[np.min(U_plot)-0.7]*len(x_p),label='x_p',lw=5)
	ax["B"].set_ylim(np.min(U_plot)-1,np.max(U_plot)+1)
	ax["B"].legend(loc='upper right')

	ax["C"].plot(x_fft,DDU,label=r'True $\Delta u$',lw=5)
	ax["C"].plot(x_fft,ddu_fft,ls='dashed',label=r'Network $\Delta u$',lw=5)
	ax["C"].legend()

	plt.suptitle('C reconstructed from noisy samples on U and PDE \n' + 
				r'$u_{lim}=$' + str(u_tol) + r' / $\delta=$' + str(delta) + 
				r' / $N_{samples}=$' + str(N_m) + r' / $N_{PDE}=$' + str(N_p) + 
				r' / $N_{fft}=$' + str(N) + '\n' +
				r'$|u-\bar{u}|_2^2=$' + str(np.mean((U_net-U_plot)**2)) + 
				r' / $|c-\bar{c}|_2^2=$' + str(np.mean((C-C_net)**2)))

	plt.show()