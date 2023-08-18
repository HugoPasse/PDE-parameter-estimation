# -*- coding: utf-8 -*-

'''
This python code trains num_net different networks on the same data and computes the fft of a vectorization of
the function given by the model. Then it plots the error made on Δu by autodifferentiation given the Wasserstein 
distance between the FFT of the model and the true FFT.

The goal of this code is to show that the higher is the error on the FFT, the higher is the error on 
the laplacian computed by autodifferentiation
'''

import numpy as np
from scipy.stats import wasserstein_distance
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from lib.problem import u_real, ddu_real
from lib.nn_siren import SirenNet
from lib.network_tools import new_net, new_optim
from lib.tools import gradient

num_net = 1000
epochs = 1000

N = 50
delta = 1e-2
x = torch.linspace(0,1,N,requires_grad=True).unsqueeze(dim=1)

U = u_real(x)
u_fft = np.fft.fft(U.flatten().detach().numpy()).real
U = U + torch.randn(N).unsqueeze(dim=1)*delta
U = U.detach()
DDU = ddu_real(x)

def train_net(net,tol=0):
	optim = new_optim(net)
	def eval_loss():
		loss = torch.nn.functional.mse_loss(net(x),U)
		optim.zero_grad()
		loss.backward()
		return loss
	for i in range(epochs):
		optim.step(eval_loss)
		if eval_loss() < tol:
			break
	return eval_loss()

def ddu_err(net):
	return float(torch.nn.functional.mse_loss(gradient(gradient(net(x),x),x),DDU))

def fft_distance(net):
	u_net = net(x).flatten().detach().numpy()
	fft_net = np.fft.fft(u_net).real
	return wasserstein_distance(fft_net,u_fft)

print("Training",num_net,"models on",epochs,"epochs")
print("N =",N,", delta =",delta)

ldduerr = []
lfftwas = []
lloss = []
tol = 1e-4
for i in range(num_net):
	print('Training network',i,'/',num_net,end='\r')
	net = new_net(dim_out=1,layers=3,neurons=16)
	lloss.append(float(train_net(net,tol)))
	ldduerr.append(ddu_err(net))
	lfftwas.append(fft_distance(net))

sc = plt.scatter(lfftwas,ldduerr,c=lloss,cmap='coolwarm')
plt.colorbar(sc)

plt.axhline(y=0, color='k',ls='dashed',linewidth=.5)

plt.ylabel('Error on the Laplacian')
plt.xlabel('Wasserstein distance between spectrums')
plt.title('Error made on the Laplacian computed by autodifferentiation\n' +
			r'$\delta=$' + '{:.0e}'.format(delta) + r' / $N=$' + str(N) + r' / $tol=$' + '{:.0e}'.format(tol))

plt.show()