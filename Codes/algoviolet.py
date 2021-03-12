"""
Kernel estimation Function
---------
      violetBD  : algorithm to compute blind deconvolution by minimization method
                 of one FBS step convexe, one FBS step convexe, one FBS concave
      FBS_im    : one FBS step for image, convexe function
      FBS_ker   : one FBS step for kernel, convexe function
      FBS_dual  : one FBS step for dual function, concave function
      Energy    : primal and dual energy for one step of algorithm

@author: Cecile Della Valle
@date: 23/02/2021
"""

# General import
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import sys
# Local import
from Codes.fbstep import FBS_ker, FBS_im, FBS_dual
from Codes.fbstep import Energy, Gradient


def violetBD(K_in,x_in,x_blurred,\
                  alpha,mu,gamma,theta=1,\
                  niter=200,coeffK=0.1,coeffx=0.1,\
                  proj_simplex=False):
    """
    one FBS step convexe, one FBS step convexe, one FBS concave
    extended chambolle-pock
       Parameters
       -----------
           K_in      (np.array) : initialisation kernel of size (2M,2M)  
           x_in      (np.array) : initial image size (Nx,Ny)
           x_blurred (np.array) : blurred and noisy image size (Nx,Ny)
           alpha        (float) : regularization parameter of Tikhopnov type for Kernel
           mu           (float) : regularization parameter for TV
           theta        (float) : relaxation parameter
           niter          (int) : number of iterations
           coeffK       (float) : coefficient on kernel FB step
           coeffx       (float) : coefficient on image FB step
           proj_simplex  (bool) : projection of Kernel over simplex space
       Returns
       ----------
           K         (np.array) : estimation of Kernel size (2M,2M)
           x         (np.array) : denoised deconvolved image size (Nx,Ny)
           EpK       (np.array) : Primal energy for kernel, size (3*niter)
           Epx       (np.array) : Primal energy for image, size (3*niter)
           Ed        (np.array) : Dual energy, size (3*niter) 
    """
    # local parameters and matrix sizes
    M,_    = K_in.shape
    M      = M//2 # kernel middle size
    Nx, Ny = x_blurred.shape # image size
    # kernel position (for padding)
    min_x  = Nx//2+1-M-2
    max_x  = Nx//2+M-1
    min_y  = Ny//2+1-M-2
    max_y  = Ny//2+M-1
    # save
    param  = M,Nx,Ny,min_x,max_x,min_y,max_y
    # initialisation
    Ep     = np.zeros(3*niter) # primal energy - Kernel and image
    Ed     = np.zeros(3*niter) # dual energy
    Ki     = K_in.copy() # kernel
    xi     = x_in.copy() # image
    xbar   = x_in.copy() # image
    xold   = x_in.copy() # image saved for relaxation
    px     = np.zeros((Nx,Ny)) # dual variable on x
    py     = np.zeros((Nx,Ny)) # dual variable on y
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.zeros((Nx,Ny))
    d_pad[Nx//2-1:Nx//2+2,Ny//2-1:Ny//2+2] = d
    # rescaling param
    regK   = alpha*(2*M)**2
    regx   = mu/Nx/Ny
    wght   = gamma
    # gradient step initial
    tauK = coeffK # primal kernel function FB step size
    taux = coeffx # primal image function FB step size 
    taup = 1 # dual function FB step size
    #
    for i in range(niter):
        tauK *=0.998
        # FBS for kernel
        Ki    = FBS_ker(xi,Ki,x_blurred,d_pad,regx,wght,tauK,param,simplex=proj_simplex)
        Ep[3*i],Ed[3*i] = Energy(xi,Ki,px,py,x_blurred,d_pad,\
                                            regK,regx,wght,param)
        # FBS for v (dual of TV)
        px,py = FBS_dual(xbar,Ki,px,py,regx,wght,taup)
        Ep[3*i+2],Ed[3*i+2] = Energy(xi,Ki,px,py,x_blurred,d_pad,\
                                                 regK,regx,wght,param)
        # FBS for image
        xi    = FBS_im(xi,Ki,px,py,x_blurred,regx,wght,taux,param)
        Ep[3*i+1],Ed[3*i+1] = Energy(xi,Ki,px,py,x_blurred,d_pad,\
                                                  regK,regx,wght,param)
        # relaxation
        xbar     = xi + theta*(xi-xold)
        xold   = xi.copy()
        # test
        if (i==0):
            gradK0,gradx0 = Gradient(xi,Ki,px,py,x_blurred,d_pad,regK,regx,wght,param)
        if (i>0):
            gradK,gradx = Gradient(xi,Ki,px,py,x_blurred,d_pad,regK,regx,wght,param)
            if i%100==0:
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                     .format(i,gradK,gradx))
            if (gradK/gradK0 <10**-3) or (gradx/gradx0<10**-3):
                print("stops at {} iterations : the algorithm converges".format(i))
                return Ki,xi,Ep,Ed
            if (Ep[3*i+1]>Ep[0]):
                print("stops prematurely at {} iterations : energy rises".format(i))
                return Ki,xi,Ep,Ed
    return Ki,xi,Ep,Ed
 
