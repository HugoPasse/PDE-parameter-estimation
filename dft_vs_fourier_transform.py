# -*- coding: utf-8 -*-

'''
This code allows us to compare the precision of the truncated Fourier differentiation method using the DFT.
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

from lib.problem import *	# You can replace this by your own functions
from lib.nn_siren import SirenNet
from lib.fourier_tools import derivative_from_fft_1d
from lib.network_tools import new_net, new_optim, lp_norm

N_m = 50 		# Number of samples on U
delta = 1e-2	# Noise on U

x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)
U = u_real(x_m) + torch.randn(N_m).unsqueeze(dim=1)*delta
U = U.detach()

def eval_loss_u():
	loss = torch.nn.functional.mse_loss(net(x_m),U)
	optim.zero_grad()
	loss.backward()
	return loss

net = new_net(dim_out=1,neurons=16,layers=3)
optim = new_optim(net)

# Training U
epochs_u = 5000
llossu = []
for i in range(epochs_u):
	optim.step(eval_loss_u)
	llossu.append(eval_loss_u())
	if llossu[-1] < max(1e-7,delta**2):
		break

x_e = torch.linspace(0,1,10000).unsqueeze(dim=1)
u_err = float(torch.nn.functional.mse_loss(net(x_e),u_real(x_e)))

print('u loss  :',float(eval_loss_u()))
print('u error :',float(u_err))

p = 100
u_norm = lp_norm(net,p,n=1000000)
mxi = 2*np.pi*(u_norm / (2*delta))**(1/p)
therr = 2*(u_norm**(2/p))*((2*delta)**(1-2/p))

print('mxsi :',mxi)

lmse = []
lmax = []
lN   = []
for N in np.logspace(2,6,100): 
	N = int(N)
	lN.append(N)
	x_fft = torch.linspace(0,1,N).unsqueeze(dim=1)

	u_fft = net(x_fft).flatten().detach().numpy()
	ddu_fft = derivative_from_fft_1d(u_fft,mxi,k=2)
	lmse.append(np.mean((ddu_real(x_fft).flatten().detach().numpy() - ddu_fft)**2))
	lmax.append(np.max((ddu_real(x_fft).flatten().detach().numpy() - ddu_fft)**2))
	
	print('N =',N)
	print('MSE on DDU       :',lmse[-1])
	print('Max error on DDU :',lmax[-1])

plt.plot(lN,lmse,label='MSE',lw=5)
plt.plot(lN,lmax,label='Max error',lw=5,ls='dotted')
plt.plot(lN,[therr]*len(lN),label='Theoretical error',lw=5,ls='dashed')

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.xlabel('Number of points used to compute the Fourier transform')
plt.ylabel('Error on the Laplacian')
plt.title(r'Error made by the truncated Fourier transform with $u = \cos(2\pi x)$' + '\n' 
			+ 'Number of points to fit u : ' + str(N_m) +  ' $\delta=${:.0e}'.format(delta) + 
			'\nNetwork error {:.4e}'.format(u_err))
plt.show()