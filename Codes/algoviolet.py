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
                  alpha,mu,gamma=1,\
                  niter=200,coeffK=1,coeffx=1,\
                  proj_simplex=False, verbose=True):
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
           gamma           (float) : weight on the data-fit term, default 1
           niter          (int) : number of iterations
           coeffK       (float) : coefficient on kernel FB step
           coeffx       (float) : coefficient on image FB step
           proj_simplex  (bool) : projection of Kernel over simplex space
           verbose       (bool) : if True, it displays intemediary result
       Returns
       ----------
           K         (np.array) : estimation of Kernel size (2M,2M)
           x         (np.array) : denoised deconvolved image size (Nx,Ny)
           Ep        (np.array) : Primal energy for image, size (3*niter)
           Ed        (np.array) : Dual energy, size (3*niter) 
    """
    # local parameters and matrix sizes
    M,_    = K_in.shape
    M      = M//2 # kernel middle size
    Nx, Ny = x_blurred.shape # image size
    # initialisation
    Ki     = K_in.copy() # kernel
    Ki     = np.pad(Ki, ((Nx//2-M,Nx//2-M),(Ny//2-M,Ny//2-M)),'constant') #padding
    #
    Ep     = np.zeros(3*niter) # primal energy - Kernel and image
    Ed     = np.zeros(3*niter) # dual energy
    xi     = x_in.copy() # image
    xbar   = x_in.copy() # image
    xold   = x_in.copy() # image saved for relaxation
    px     = np.zeros((Nx,Ny)) # dual variable on x
    py     = np.zeros((Nx,Ny)) # dual variable on y
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.pad(d, ((Nx//2-1,Nx//2-1),(Ny//2-1,Ny//2-1)), 'constant')
    # gradient step initial
    tauK   = coeffK # primal kernel function FB step size coeff
    taux   = coeffx/2 # primal image function FB step size coeff 
    taup   = coeffx/2 # dual function FB step size coeff
    theta  = 1 # relaxation parameter
    wght   = gamma
    #
    for i in range(niter):
        # tauK *=0.998
        # FBS for kernel
        Ki                  = FBS_ker(xi,Ki,x_blurred,d_pad,alpha,gamma=wght,coeff=tauK,simplex=proj_simplex)
        Ep[3*i],Ed[3*i]     = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu)
        # FBS for image
        xi                  = FBS_im(xi,Ki,px,py,x_blurred,mu,gamma=wght,coeff=taux)
        Ep[3*i+1],Ed[3*i+1] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
        # FBS for v (dual of TV)
        px,py               = FBS_dual(xbar,px,py,mu,gamma=wght,coeff=taup)
        Ep[3*i+2],Ed[3*i+2] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
        # relaxation
        xbar  = xi + theta*(xi-xold)
        xold  = xi.copy()
        # test
        if (i==0):
            gradK0,gradx0   = Gradient(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
        if (i>0):
            gradK,gradx     = Gradient(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
            stat = niter//50
            if (i%stat==0)&(verbose):
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                     .format(i,gradK,gradx))
            # Test gradient and energy
            if (gradK/gradK0 <10**-10) or (gradx/gradx0<10**-3):
                print("stops at {} iterations : the algorithm converges".format(i))
                Ki = Ki[Nx//2-M:Nx//2+M+1,Ny//2-M:Ny//2+M+1]
                return Ki,xi,Ep,Ed
            elif (coeffK>0)and(gradK/gradK0 >100):
                print("stops at {} iterations : gradient of K rises".format(i))
                Ki = Ki[Nx//2-M:Nx//2+M+1,Ny//2-M:Ny//2+M+1]
                return Ki,xi,Ep,Ed
            elif (gradx/gradx0>100) :
                print("stops at {} iterations : gradient of image rises".format(i))
                Ki = Ki[Nx//2-M:Nx//2+M+1,Ny//2-M:Ny//2+M+1]
                return Ki,xi,Ep,Ed
            elif (Ep[3*i+1]>10*Ep[0]):
                print("stops prematurely at {} iterations : energy rises".format(i))
                Ki = Ki[Nx//2-M:Nx//2+M+1,Ny//2-M:Ny//2+M+1]
                return Ki,xi,Ep,Ed
    # retrun
    print('Final energy :',Ep[-1])
    Ki = Ki[Nx//2-M:Nx//2+M+1,Ny//2-M:Ny//2+M+1]
    return Ki,xi,Ep,Ed
 
