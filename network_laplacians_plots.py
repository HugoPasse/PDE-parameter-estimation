# -*- coding: utf-8 -*-

'''
This python script trains a network on a function u, then it computes Δu with three different methods :
autodifferentiation, finite differences, truncated fft
It plots the four Δu (the real one and the three approxes) and the pointwise differences
'''

import numpy as np
import torch
import matplotlib.pyplot as plt

from lib.problem import u_real, ddu_real
from lib.nn_siren import SirenNet
from lib.fourier_tools import derivative_from_fft_1d
from lib.network_tools import new_net, new_optim
from lib.tools import finite_difference_laplacian_1d, gradient

N_m = 30
delta = 1e-2
x_m = torch.linspace(0,1,N_m,requires_grad=True).unsqueeze(dim=1)
U = u_real(x_m) + torch.randn(N_m).unsqueeze(dim=1)*delta
U = U.detach()

net = new_net(dim_out=1, neurons=16, layers=3)
optim = new_optim(net)

def eval_loss():
	loss  = torch.nn.functional.mse_loss(net(x_m),U)
	optim.zero_grad()
	loss.backward()
	return loss

epochs = 5000
lloss = []
for i in range(epochs):
	lloss.append(float(eval_loss()))
	optim.step(eval_loss)

print(lloss[-1])

N = 4096
x = torch.linspace(0,1,N,requires_grad=True).unsqueeze(dim=1)

# Auto-differentiation
autod_ddu = gradient(gradient(net(x),x),x).flatten().detach().numpy()
# Finites differences
fd_ddu = finite_difference_laplacian_1d(net,x,lloss[-1]**0.5).flatten().detach().numpy()
# Truncated Fourier transform
tfft_ddu = derivative_from_fft_1d(net(x).flatten().detach().numpy(),mxsi=20,k=2)
# True value of the Laplacian
true_ddu = ddu_real(x).flatten().detach().numpy()

x = x.flatten().detach().numpy()

fig,ax = plt.subplot_mosaic("AAB;AAC;AAD")

ax["A"].plot(x,true_ddu,lw=3,label=r'True $\Delta u$')
ax["A"].plot(x,autod_ddu,lw=2,ls='dashed',label=r'Auto-differentation $\Delta u$')
ax["A"].plot(x,fd_ddu,lw=2,ls='dashed',label=r'Finite differences $\Delta u$')
ax["A"].plot(x,tfft_ddu,lw=2,ls='dashed',label=r'Truncated FFT $\Delta u$')
ax["A"].legend()
ax["A"].set_title(r'$\Delta u$ ' + 'approximation by method\n' 
				+ r'$N_m$ = ' + str(N_m) + ' / N = ' + str(N) 
				+ r' / $\delta$ = ' + str(delta) + r' / loss$_u$ = ' + str(lloss[-1]))

ax["B"].plot(x,np.abs(true_ddu-autod_ddu),label=r'$|\Delta u - \Delta u_{auto}|$')
ax["B"].legend()
ax["B"].set_title(r'$|\Delta u - \Delta u_{auto}|_2^2$ = ' + str(np.mean((true_ddu-autod_ddu)**2)))

ax["C"].plot(x,np.abs(true_ddu-fd_ddu),label=r'$|\Delta u - \Delta u_{fd}|$')
ax["C"].legend()
ax["C"].set_title(r'$|\Delta u - \Delta u_{fd}|_2^2$ = ' + str(np.mean((true_ddu-fd_ddu)**2)))

ax["D"].plot(x,np.abs(true_ddu-tfft_ddu),label=r'$|\Delta u - \Delta u_{fft}|$')
ax["D"].legend()
ax["D"].set_title(r'$|\Delta u - \Delta u_{fft}|_2^2$ = ' + str(np.mean((true_ddu-tfft_ddu)**2)))

plt.subplots_adjust(top=0.88,
					bottom=0.11,
					left=0.125,
					right=0.9,
					hspace=0.35,
					wspace=0.2)

plt.suptitle("Comparison between different methods to compute " + r'$\Delta u$')
plt.show()

# 3 methods : auto-diff, finite-differences, truncated Fourier method.
# TODO : implement them