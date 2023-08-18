 # -*- coding: utf-8 -*-

import torch
import numpy as np

def c_real(x):
    return np.sqrt(2)*torch.cos(2*np.pi*x) + torch.cos(4*np.pi*x)

def u_real(x):
    return torch.cos(2*np.pi*x)

def ddu_real(x):
    return -4*np.pi**2*torch.cos(2*np.pi*x)

def phi_real(x):
	return -ddu_real(x) + c_real(x)*u_real(x)