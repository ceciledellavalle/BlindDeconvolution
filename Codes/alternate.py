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
    # kernel position (for padding)
    min_x  = Nx//2+1-M-2
    max_x  = Nx//2+M-1
    min_y  = Ny//2+1-M-2
    max_y  = Ny//2+M-1
    # save
    dict_param  = M,Nx,Ny,min_x,max_x,min_y,max_y # dictionnary of parameters
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.zeros((Nx,Ny))
    d_pad[Nx//2-1:Nx//2+2,Ny//2-1:Ny//2+2] = d
    # initialisation
    Etot   = np.zeros(alte*niter_Lap+2*alte*niter_TV)
    Ki     = K_in.copy()
    xi     = x_in.copy()
    xi     = x_in.copy() # image
    xbar   = x_in.copy() # image
    xold   = x_in.copy() # image saved for relaxation
    px     = np.zeros((Nx,Ny)) 
    py     = np.zeros((Nx,Ny))
    # rescaling
    regK     = alpha*(2*M)**2
    regx     = mu/Nx/Ny
    for i in range(alte):
        # First estimation of image
        print('------------- min image -----------------')
        for n in range(niter_TV):
            # one FBS for image
            xi    = FBS_im(xi,Ki,px,py,x_blurred,regx,1,0.99,dict_param)
            # energy
            Etot[i*(niter_Lap+2*niter_TV)+2*n],_ = Energy(xi,Ki,px,py,x_blurred,\
                                              d_pad,regK,regx,1,dict_param)
            # one FBS for dual variable
            px,py = FBS_dual(xbar,Ki,px,py,regx,1,0.99)
            # energy
            Etot[i*(niter_Lap+2*niter_TV)+2*n+1],_ = Energy(xi,Ki,px,py,x_blurred,\
                                              d_pad,regK,regx,1,dict_param)
            # relaxation
            xbar  = 2*xi-xold
            xold  = xi.copy()
            # test
            counter = i*(niter_Lap+2*niter_TV)+2*n
            if counter%500==0:
                gradK,gradx = Gradient(xi,Ki,px,py,x_blurred,d_pad,regK,regx,1,dict_param)
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                         .format(counter,gradK,gradx))
        # Second estimation of Kernel
        print('------------- min kernel -----------------')
        for m in range(niter_Lap):
            # one FBS for kernel
            Ki   = FBS_ker(xi,Ki,x_blurred,d_pad,regK,1,1,dict_param,simplex=False)
            # energy
            Etot[(i+1)*2*niter_TV+i*niter_Lap+m],_ = Energy(xi,Ki,px,py,x_blurred,\
                                              d_pad,regK,regx,1,dict_param)
            # test
            counter = (i+1)*2*niter_TV+i*niter_Lap+m
            if counter%500==0:
                gradK,gradx = Gradient(xi,Ki,px,py,x_blurred,d_pad,regK,regx,1,dict_param)
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                         .format(counter,gradK,gradx))
    return Ki,xi,Etot
        
