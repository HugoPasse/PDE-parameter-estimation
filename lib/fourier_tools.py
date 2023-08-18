 # -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import diags

# Computes the diagonal matrix used for truncation
def diagonal_interpolant(N,mxsi,k=1):
    """
    N    : size of the matrix to return
    mxsi : limit for truncation
    k    : order of the derivative  
    """
    M = np.arange(N) * (2*np.pi)
    for i in range(N):
        if np.abs(M[i]) > mxsi:
            M[i] = 0
        else:
            M[i] = M[i]**k
    return diags([M*(1J)**k],[0])

# Approximates the derivate of a 1-d function from samples using FFT truncation
def derivative_from_fft_1d(F,mxsi,k):
    """
    F    : array of the sample points
    mxsi : limit for truncation
    k    : order of the derivative 
    """
    N = len(F)
    A = diagonal_interpolant(N,mxsi,k)
    return 2*np.real(np.fft.ifft(A.dot(np.fft.fft(F))))

# Approximates the derivate of a 2-d function from samples using FFT truncation
def derivative_from_fft_2d(F,mxsi,k):
    """
    F    : array of the sample points
    mxsi : limit for truncation
    k    : order of the derivative 
    """
    N = len(F)
    A = diagonal_interpolant(N,mxsi,k)
    return 2*np.real(np.fft.ifft(np.linalg.matrix_power(A,k)).dot(np.fft.fft(F)))

def laplacian_from_fft_2d(F,mxsi):
    """
    F    : array of the sample points
    mxsi : limit for truncation
    """
    M = diagonal_interpolant(F.shape[0],mxsi,k=2)
    fft_f = np.fft.fft2(F)
    ddu_x = np.fft.ifft2(M.dot(fft_f)).real
    ddu_y = np.fft.ifft2((M.dot(fft_f.T)).T).real
    return (ddu_x + ddu_y) / 2