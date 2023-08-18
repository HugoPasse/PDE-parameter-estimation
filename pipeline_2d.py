# -*- coding: utf-8 -*-

'''
This is an implementation of the final method for bivariate functions
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
plt.rcParams.update({'font.size': 14})

from lib.nn_siren import SirenNet
from lib.fourier_tools import laplacian_from_fft_2d
from lib.network_tools import new_net, new_optim, num_param

def c_real(x,y):
	return torch.cos(np.pi*x) + torch.sin(np.pi*y)

def u_real(x,y):
	return torch.cos(2*np.pi*(x+y))

def ddu_real(x,y):
	return -8*np.pi**2*torch.cos(2*np.pi*(x+y))

def phi_real(x,y):
	return -ddu_real(x,y) + u_real(x,y)*c_real(x,y)

N = 25
delta = 1e-3
xs = torch.linspace(-1,1,N)
ys = torch.linspace(-1,1,N)
x,y = torch.meshgrid(xs, ys, indexing='xy')

pts = torch.cat((x.flatten().unsqueeze(dim=1), y.flatten().unsqueeze(dim=1)),dim=1)

U = u_real(x,y) + torch.randn(N*N).reshape(N,N)*delta
U = U.flatten().detach()

def train_u(epochs = 1000):
	optim = new_optim(net_u)
	
	def eval_loss():
		u_net = net_u(pts).flatten()
		loss = torch.nn.functional.mse_loss(u_net,U)
		optim.zero_grad()
		loss.backward()
		return loss
	
	for i in range(epochs):
		optim.step(eval_loss)
	return eval_loss()

net_u = new_net(dim_in=2,dim_out=1)
print("u loss =",float(train_u()))

# Computing gradient
Ng = 1000
xsg = torch.linspace(-1,1,Ng)
ysg = torch.linspace(-1,1,Ng)
xg,yg = torch.meshgrid(xsg, ysg, indexing='xy')
ptsg = torch.cat((xg.flatten().unsqueeze(dim=1), yg.flatten().unsqueeze(dim=1)),dim=1)
ug = net_u(ptsg).reshape(Ng,Ng).detach()
DDU = ddu_real(xg,yg).flatten()
ddu_net = torch.tensor(laplacian_from_fft_2d(ug.numpy(),mxsi=15)).flatten().detach()
PHI = phi_real(xg,yg).flatten().detach()
ug = ug.flatten()

# Training c
def train_c(epochs = 100):
	optim = new_optim(net_c)
	
	def eval_loss():
		c_net = net_c(ptsg).flatten()
		phi_net = -ddu_net + ug*c_net
		loss = torch.nn.functional.mse_loss(phi_net.float(),PHI)
		optim.zero_grad()
		loss.backward()
		return loss
	
	for i in range(epochs):
		optim.step(eval_loss)
	return eval_loss()

net_c = new_net(dim_in=2,dim_out=1)
print('C loss :',float(train_c()))

# Errors
Ne = 1000
xse = torch.linspace(-1,1,Ne)
yse = torch.linspace(-1,1,Ne)
xe,ye = torch.meshgrid(xse, yse, indexing='xy')
ptse = torch.cat((xe.flatten().unsqueeze(dim=1), ye.flatten().unsqueeze(dim=1)),dim=1)
U = u_real(xe,ye).flatten().detach()
u_net = net_u(ptse).flatten().detach()

print("Mse on U         :", float(torch.nn.functional.mse_loss(u_net,U)))
print("Mse on DDU       :", float(torch.nn.functional.mse_loss(ddu_net,DDU)))
print("Max error on DDU :", float(np.max((ddu_net.flatten().detach().numpy()-DDU.flatten().detach().numpy())**2)))


# Plots
Np = 377
xsp = torch.linspace(-1,1,Np)
ysp = torch.linspace(-1,1,Np)
xp,yp = torch.meshgrid(xsp, ysp, indexing='xy')
ptsp = torch.cat((xp.flatten().unsqueeze(dim=1), yp.flatten().unsqueeze(dim=1)),dim=1)

C = c_real(xp,yp)
c_net = net_c(ptsp).reshape(Np,Np)

print("Mse on c         :", float(torch.nn.functional.mse_loss(C,c_net)))

C = C.detach().numpy()
c_net = c_net.detach().numpy()
diff = np.abs(C - c_net)

print("Max error on c   :", np.max((C-c_net)**2))

fig, ax = plt.subplot_mosaic("ACC;BCC",subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)

ax["A"].plot_surface(xp, yp, C, color='b', lw=0.5, rstride=8, cstride=8, alpha=0.3)
ax["A"].set_title('True c')

ax["B"].plot_surface(xp, yp, c_net, color='r', lw=0.5, rstride=8, cstride=8, alpha=0.3)
ax["B"].set_title('Network c')

ax["C"].plot_surface(xp, yp, C, color='b', lw=0.5, rstride=8, cstride=8, alpha=0.3)
ax["C"].plot_surface(xp, yp, c_net, color='r', lw=0.5, rstride=8, cstride=8, alpha=0.3)
cont = ax["C"].contourf(xp,yp,diff, zdir='z',offset=-2, cmap='coolwarm')
ax["C"].set_title(r'True c, network c and $\left\| c-\bar{c}\right\|_2$')


fig.colorbar(cont,extend='both',location='bottom')
fig.suptitle('C reconstructed from noisy samples on U and PDE \n' +
			r'$\delta=$' + str(delta) + r' / $N_m=$' + str(N**2) + 
			r' / $N_{fft}=$' + str(Ng**2))
plt.show()