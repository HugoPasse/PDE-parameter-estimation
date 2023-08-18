# -*- coding: utf-8 -*-
# This code is from Mattéo Clémot
# http://perso.ens-lyon.fr/matteo.clemot/

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def laplacian2d(f, x):  
    g = gradient(f(x), x)
    h0 = gradient(g[:,0], x)
    h1 = gradient(g[:,1], x)
    return h0[:,0] + h1[:,1]

def hessian2d(f, x):
    g = gradient(f(x), x)
    h0 = gradient(g[:,0], x)[:,None,:]
    h1 = gradient(g[:,1], x)[:,None,:]
    h = torch.cat((h0,h1), dim=1)
    return h

def disc_laplacian(n):
    B = torch.diag_embed(-4*torch.ones(n)) + torch.diag_embed(torch.ones(n-1), offset=1) + torch.diag_embed(torch.ones(n-1), offset=-1)
    L = B
    for _ in range(n-1):
        L = torch.block_diag(L, B)
    L += torch.diag_embed(torch.ones(n*(n-1)), offset=n)
    L += torch.diag_embed(torch.ones(n*(n-1)), offset=-n)
    L *= (n-1)**2
    return L.to(device)

def disc_laplacian1d(n):
    L = torch.diag_embed(-2*torch.ones(n)) + torch.diag_embed(torch.ones(n-1), offset=1) + torch.diag_embed(torch.ones(n-1), offset=-1)
    L *= (n-1)**2
    return L.to(device)

def condition(f, x):
    fx = f(x)
    g = gradient(fx, x)
    res = torch.linalg.norm(g, dim=1) * torch.linalg.norm(x, dim=1) / fx.flatten()
    return res

# The code below is mine

def finite_difference_laplacian_1d(net,x,err):
    h2 = err**2
    net_x = net(x)[:,0]
    net_left = net(x+h2**0.5)[:,0]
    net_right = net(x-h2**0.5)[:,0]
    return (net_right + net_left - 2*net_x)/h2 