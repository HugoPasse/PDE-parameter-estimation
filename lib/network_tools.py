 # -*- coding: utf-8 -*-

import torch
import numpy as np

from lib.nn_siren import SirenNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Returns a new network with the specified architecture
def new_net(dim_out=1,layers=3,neurons=16,dim_in=1):
    net = SirenNet(
        dim_in = dim_in,
        dim_hidden = neurons,
        dim_out = dim_out,
        num_layers = layers,
        activation = None,
        w0_initial = 10.,
        w0 = 10.,
        ).to(device)
    return net

# Returns a new Adam optimizer
def new_optim(net,lr=8e-4):
    optim = torch.optim.Adam(lr=8e-4, params=net.parameters())
    return optim

# Updates the parameters of the network
def update_param(net, d):
    i = 0
    for p in net.parameters():
        p.data += d[i:i+torch.numel(p.data)].reshape(p.data.shape)
        i += torch.numel(p.data)

# Adds d to the parameter at index idx in net
def update_single_param(net,idx,d):
    for p in net.parameters():
        cnt = torch.numel(p.data)
        if idx < cnt:
            sh = p.data.shape
            tmp = torch.flatten(p.data)
            tmp[idx] = tmp[idx]+d
            p.data = torch.reshape(tmp,sh)
            return
        else:
            idx -= cnt

# Computes the norm of the parameters vector
def nn_norm(net):
    W = []
    for name, param in net.named_parameters():
        W += param.data.flatten().detach().tolist()
    return np.linalg.norm(np.array(W))

# Returns the number of parameters in the network
def num_param(net):
    i = 0
    for p in net.parameters():
        i += torch.numel(p.data)
    return i

# Returns an approximate of the Lp norm of the model
def lp_norm(net,p,n=10000):
    # n is the number of points used for the integral computation
    x = torch.linspace(0,1,n).unsqueeze(dim=1)
    f = np.abs(net(x).flatten().detach().numpy()) ** p
    res = 1/n * 0.5 * (2*np.sum(f) - f[0] - f[-1])
    return res**(1/p)