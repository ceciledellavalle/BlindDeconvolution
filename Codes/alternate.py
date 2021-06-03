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


def AlternatingBD(K_in,x_in,x_blurred,alpha,mu,gamma=1,\
                  alte=4,niter_TV=500,niter_Lap =500,\
                  proj_simplex=False,verbose=True):
    """
    Alternating estimation of a blind deconvolution.
    Parameters
    ----------
        K_in      (numpy array) : initial kernel of size (2M,2M)
        x_in     (numpy array) : initial image of size (Nx,Ny)
        x_blurred (numpy array) : blurred and noisy image of size (Nx,Ny)
        alpha           (float) : regularisation parameter for Tikhonov type regularization
        mu              (float) : regularisation parameter for TV
        gamma           (float) : weight on the data-fit term, default 1
        alte              (int) : number of alternating iterations
        niter_TV          (int) : number of iterations for image reconstruction
        niter_TV          (int) : number of iterations for kernel reconstruction
        proj_simplex     (bool) : if True, then projection of kernel on simplex space
        verbose          (bool) : if True, it displays intemediary result
    Returns
    --------
        Ki      (numpy array) : final estimated kernel of size (2M,2M)
        xi      (numpy array) : final deblurred denoised image of size (Nx,Ny)
        Ep      (numpy array) : primal energy through every FB step
        Ed      (numpy.array) : Dual energy
    """
    # local parameters and matrix sizes
    M,_    = K_in.shape
    M      = M//2 # kernel middle size
    Nx, Ny = x_blurred.shape # image size
    # initialisation
    Ki     = K_in.copy() # kernel
    Ki     = np.pad(Ki, ((Nx//2-M,Nx//2-M),(Ny//2-M,Ny//2-M)),'constant') #padding
    Ep,Ed  = np.zeros(alte*niter_Lap+2*alte*niter_TV),np.zeros(alte*niter_Lap+2*alte*niter_TV)
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
    wght   = gamma
    # initialisation
    Ep,Ed  = np.zeros(alte*niter_Lap+2*alte*niter_TV),np.zeros(alte*niter_Lap+2*alte*niter_TV)
    #
    count  = 0
    for i in range(alte):
        # First estimation of image
        if verbose:
            print('------------- min image -----------------')
        for n in range(niter_TV):
            # one FBS for dual variable
            px,py               = FBS_dual(xbar,px,py,mu,gamma=wght)
            # energy
            Ep[count],Ed[count] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
            count              +=1
            # one FBS for image
            xi                  = FBS_im(xi,Ki,px,py,x_blurred,mu,gamma=wght)
            # energy
            Ep[count],Ed[count] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
            count              +=1
            # relaxation
            xbar                = 2*xi-xold
            xold                = xi.copy()
            # test
            counter             = i*(niter_Lap+2*niter_TV)+2*n
            if (counter%500==0)&(verbose):
                gradK,gradx = Gradient(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                         .format(counter,gradK,gradx))
        print("Energie = ",Ep[count-1])
        # Second estimation of Kernel
        if verbose:
            print('------------- min kernel -----------------')
        for m in range(niter_Lap):
            # one FBS for kernel
            Ki                  = FBS_ker(xi,Ki,x_blurred,d_pad,alpha,gamma=wght,simplex=proj_simplex)
            # energy
            Ep[count],Ed[count] = Energy(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
            count              += 1
            # test
            counter = (i+1)*2*niter_TV+i*niter_Lap+m
            if (counter%500==0)&(verbose):
                gradK,gradx = Gradient(xi,Ki,px,py,x_blurred,d_pad,alpha,mu,gamma=wght)
                print("iteration {} %--- gradient K {:.4f} --- gradient x {:.4f}"\
                         .format(counter,gradK,gradx))
        print("Energie = ",Ep[count-1])
    # retrun
    print('Final energy :',Ep[-1])
    Ki = Ki[Nx//2-M:Nx//2+M+1,Ny//2-M:Ny//2+M+1]
    return Ki,xi,Ep,Ed
        
