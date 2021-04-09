"""
Kernel estimation Function
---------
    AlternatingBD   : algorithm to compute blind deconvolution by alternating minimization method
                      Chambolle-Pock algorithm for deblurring denoising
                      FB algorithm with regularization of Tikhonov type to estimate a convolution kernel

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
from Codes.display import Display_ker
from Codes.display import Display_im


def AlternatingBD(K_in,x_in,x_blurred,alpha,mu,\
                  alte=10,niter_TV=200,niter_Lap =200,\
                  proj_simplex=False):
    """
    Alternating estimation of a blind deconvolution.
    Parameters
    ----------
        K_in      (numpy array) : initial kernel of size (2M,2M)
        x_in     (numpy array) : initial image of size (Nx,Ny)
        x_blurred (numpy array) : blurred and noisy image of size (Nx,Ny)
        alpha           (float) : regularisation parameter for Tikhonov type regularization
        mu              (float) : regularisation parameter for TV
        alte              (int) : number of alternating iterations
        niter_TV          (int) : number of iterations for image reconstruction
        niter_TV          (int) : number of iterations for kernel reconstruction
        proj_simplex     (bool) : boolean, True if projection of kernel on simplex space
    Returns
    --------
        Ki      (numpy array) : final estimated kernel of size (2M,2M)
        xi      (numpy array) : final deblurred denoised image of size (Nx,Ny)
        Etot    (numpy array) : primal energy through every FB step
    """
    # local parameters and matrix sizes
    M,_    = K_in.shape
    M      = M//2 # kernel middle size
    Nx, Ny = x_blurred.shape # image size
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.zeros((Nx,Ny))
    d_pad[Nx//2-1:Nx//2+2,Ny//2-1:Ny//2+2] = d
    # initialisation
    Ep,Ed  = np.zeros(alte*niter_Lap+2*alte*niter_TV),np.zeros(alte*niter_Lap+2*alte*niter_TV)
    Ki     = K_in.copy()
    xi     = x_in.copy()
    xi     = x_in.copy() # image
    xbar   = x_in.copy() # image
    xold   = x_in.copy() # image saved for relaxation
    px     = np.zeros((Nx,Ny)) 
    py     = np.zeros((Nx,Ny))
    #
    count  = 0
    for i in range(alte):
        # First estimation of image
        print('------------- min image -----------------')
        for n in range(niter_TV):
            # one FBS for image
            xi                  = FBS_im(xi,Ki,px,py,x_blurred,mu)
            # energy
            Ep[count],Ed[count] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu)
            count              +=1
            # one FBS for dual variable
            px,py               = FBS_dual(xbar,Ki,px,py,mu)
            # energy
            Ep[count],Ed[count] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu)
            count              +=1
            # relaxation
            xbar                = 2*xi-xold
            xold                = xi.copy()
            # test
            counter             = i*(niter_Lap+2*niter_TV)+2*n
            if counter%500==0:
                gradK,gradx = Gradient(xi,Ki,px,py,x_blurred,d_pad,regK,regx,1,dict_param)
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                         .format(counter,gradK,gradx))
        # Second estimation of Kernel
        print('------------- min kernel -----------------')
        for m in range(niter_Lap):
            # one FBS for kernel
            Ki                  = FBS_ker(xi,Ki,x_blurred,d_pad,alpha,simplex=proj_simplex)
            # energy
            Ep[count],Ed[count] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu)
            count              += 1
            # test
            counter = (i+1)*2*niter_TV+i*niter_Lap+m
            if counter%500==0:
                gradK,gradx = Gradient(xi,Ki,px,py,x_blurred,d_pad,alpha,mu)
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                         .format(counter,gradK,gradx))
    return Ki,xi,Ep,Ed
        
