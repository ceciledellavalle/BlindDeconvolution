"""
Function used in the blind deconvolution algorithm
---------
      convolve  : 2D convolution of matrices of same size, using FFT
	  projB     : projection of 2 vectors in the infty ball of a defined radius
	  nablah    : gradient
	  divh      : divergence (gradient dual)

@author: Cecile Della Valle
@date: 23/02/2021
"""
# General import
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import math

# convolution 2d
def convolve(u,v):
    """
    Convolution of 2D vectors of same size.
    """
    return fftconvolve(u,v,mode='same')
    
# function projection on B(alpha)
def projB(px,py,mu) :
    """
    Projection on ball of radiux mu.
    """
    norm      = np.sqrt(px**2+py**2)
    cond      = norm>mu
    ppx, ppy = px.copy(), py.copy()
    ppx[cond] = mu*px[cond]/norm[cond]
    ppy[cond] = mu*py[cond]/norm[cond]
    return ppx, ppy

# nablah
def nablah(v):
    """
    Gradient of (Nx,Ny) size vectors.
    """
    Nx,Ny    = v.shape
    ddx, ddy = np.zeros((Nx,Ny)), np.zeros((Nx,Ny))
    ddx[:-1,:] = v[1:,:]-v[:-1,:]
    ddy[:,:-1] = v[:,1:]-v[:,:-1]
    return ddx, ddy 
    
# divh
def divh(px,py):
    """
    Divergence of (Nx,Ny)^2 vectors, dual of gradient.
    """
    Nx,Ny      = px.shape
    dd         = np.zeros((Nx,Ny))
    dd[0,:]   += -px[0,:]
    dd[1:-1,:]+= px[:-2,:]-px[1:-1,:]
    dd[-1,:]  += px[-2,:]
    dd[:,0]   += -py[:,0]
    dd[:,1:-1]+= py[:,:-2]-py[:,1:-1]
    dd[:,-1]  += py[:,-2]
    return dd
    
