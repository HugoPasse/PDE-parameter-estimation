# -*- coding: utf-8 -*-

'''
This code computes the mean square errors made on Δu by the different methods compared to the noise on the data.
'''

import numpy as np
import torch
import matplotlib.pyplot as plt

from lib.problem import u_real, ddu_real
from lib.nn_siren import SirenNet
from lib.fourier_tools import derivative_from_fft_1d
from lib.network_tools import new_net, new_optim, lp_norm
from lib.tools import finite_difference_laplacian_1d, gradient


def eval_loss():
	loss  = torch.nn.functional.mse_loss(net(x_m),U)
	optim.zero_grad()
	loss.backward()
	return loss

def train_net(net,optim,epochs):
	for i in range(epochs):
		optim.step(eval_loss)
	return float(eval_loss())

def compute_laplacians_errors(net,delta):
	true_ddu = ddu_real(x).flatten().detach().numpy()

	autod_ddu = gradient(gradient(net(x),x),x).flatten().detach().numpy()
	autod_err = np.mean((true_ddu-autod_ddu)**2)

	fd_ddu = finite_difference_laplacian_1d(net,x,lloss[-1][-1]**0.5).flatten().detach().numpy()
	fd_err = np.mean((true_ddu-fd_ddu)**2)

	p = 100
	u_norm = lp_norm(net,p,n=1000000)
	idealxi = 2*np.pi*(u_norm / (2*delta))**(1/p)
	tfft_ddu = derivative_from_fft_1d(net(x).flatten().detach().numpy(),mxsi=idealxi,k=2)
	tfft_err = np.mean((true_ddu-tfft_ddu)**2)

	return autod_err,fd_err,tfft_err

# Number of points to train the network
N_m = 30
x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)

# Number of points to compute the laplacians
N = 100000
x = torch.linspace(0,1,N,requires_grad=True).unsqueeze(dim=1)

# Errors on the data
deltas = [10**i for i in range(-8,0)]

trains = 10
epochs = 100
lloss = []

lautoerr = []
lfderr = []
ltffterr = []

for d in deltas:
	lloss.append([])
	lautoerr.append([])
	lfderr.append([])
	ltffterr.append([])
	for i in range(trains):
		print('Training net n° ' + str(i+1) + ' for delta = ' +str(d))
		U = u_real(x_m) + torch.randn(N_m).unsqueeze(dim=1)*d
		U = U.detach()

		net = new_net(dim_out=1,neurons=16,layers=3)
		optim = new_optim(net)
		
		loss = train_net(net,optim,5000)
		lloss[-1].append(loss)
		errs = compute_laplacians_errors(net,d)
		lautoerr[-1].append(errs[0])
		lfderr[-1].append(errs[1])
		ltffterr[-1].append(errs[2])

lloss = np.mean(lloss,axis=1)
lautoerr = np.mean(lautoerr,axis=1)
lfderr = np.mean(lfderr,axis=1)
ltffterr = np.mean(ltffterr,axis=1)

print('Training loss              :',lloss)
print('Auto-differentiation error :',lautoerr)
print('Finite-differences loss    :',lfderr)
print('Truncated FFT loss         :',ltffterr)