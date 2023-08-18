# -*- coding: utf-8 -*-


'''
This script fits networks with different architectures on samples of u to see the impact of the architecture on the DFT

'''

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from lib.problem import u_real, ddu_real
from lib.nn_siren import SirenNet
from lib.fourier_tools import derivative_from_fft_1d
from lib.network_tools import new_net, new_optim, num_param
from lib.tools import finite_difference_laplacian_1d, gradient

N_m = 100
delta = 1e-2
x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)
U = u_real(x_m) + torch.randn(N_m).unsqueeze(dim=1)*delta
U = U.detach()

N = 4096
x = torch.linspace(0,1,N,requires_grad=True).unsqueeze(dim=1)

def compute_laplacians_errors(net):
	true_ddu = ddu_real(x).flatten().detach().numpy()

	autod_ddu = gradient(gradient(net(x),x),x).flatten().detach().numpy()
	autod_err = np.mean((true_ddu-autod_ddu)**2)

	fd_ddu = finite_difference_laplacian_1d(net,x,lloss[-1]**0.5).flatten().detach().numpy()
	fd_err = np.mean((true_ddu-fd_ddu)**2)

	tfft_ddu = derivative_from_fft_1d(net(x).flatten().detach().numpy(),mxsi=20,k=2)
	tfft_err = np.mean((true_ddu-tfft_ddu)**2)

	return autod_err,fd_err,tfft_err

def eval_loss():
	loss  = torch.nn.functional.mse_loss(net(x_m),U)
	optim.zero_grad()
	loss.backward()
	return loss

def train_net(tol=0):
	epochs = 2500
	for i in range(epochs):
		optim.step(eval_loss)
		if eval_loss() < tol:
			break
	return float(eval_loss())

lneurons = [j*10**i for i in range(1,3) for j in range(1,10)]
llayers = [3]
n_train = 20

num_net = n_train*len(llayers)*len(lneurons)

lloss = []
ddu_errors = [[],[],[]]
lnparams = []

for neurons in lneurons:
	for layers in llayers:
		for k in range(n_train):
			net = new_net(dim_out=1, neurons=neurons, layers=layers)
			optim = new_optim(net)
			lnparams.append(num_param(net))
			lloss.append(train_net(tol=delta**2))
			errors = compute_laplacians_errors(net)
			for i in range(3):
				ddu_errors[i].append(errors[i])
		print('Successfully trained',n_train,'networks with',layers,'layers and',neurons,'neurons')

fig,ax = plt.subplot_mosaic("A;B")

scat = ax["A"].scatter(lnparams,ddu_errors[0],c=lloss,cmap='coolwarm',marker='+',label='Auto-differentiation')
scat = ax["B"].scatter(lnparams,ddu_errors[1],c=lloss,cmap='coolwarm',marker='+',label='Finite-differences')
# ax["C"].scatter(lnparams,ddu_errors[2],c=lloss,cmap='coolwarm',marker='+',label='Truncated FFT')

for i in ["A","B"]:
	ax[i].set_xscale('log')
	ax[i].set_yscale('log')
	ax[i].set_ylabel(r'MSE on $\Delta u$')
	ax[i].set_xlabel('Number of parameters',loc='left')
	ax[i].legend()

cbar = fig.colorbar(scat,extend='both',location='bottom')
cbar.set_label("Network loss")
fig.suptitle('Error on the Laplacian given the number of parameters with various methods\n' + 
				r'$N_m=$' + str(N_m) + r' / $\delta=$' + str(delta) +
				'\n Total number of networks : ' + str(num_net))

plt.show()