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
from Codes.simplex import Simplex
from Codes.myfunc import convolve
from Codes.myfunc import projB
from Codes.myfunc import nablah
from Codes.myfunc import divh


# One forward-bakward step for the kernel
def FBS_ker(u,K_pad,g,d_pad,alpha,gamma=1,coeff=1,simplex=False):
    """
    one step of forward-backward algorithm for the image varaible
    Parameters
    ----------
        u       (np.array) : image variable of size (Nx,Ny)
        K_pad   (np.array) : padded kernel variable of size (Nx,Ny)
        g       (np.array) : noisy blurred image of size (Nx,Ny)
        d_pad   (np.array) : derivarion for regularization
        alpha      (float) : parameter for kernel of regularization of Tikhonov type 
        gamma      (float) : weight on the datafit term
        coeff      (float) : coefficient on the gradient descent step
        simplex     (bool) : boolean, True if projection on simplex applied for kernel
    Returns
    -------
        Kk     (np.array) : kernel variable after one FB step, of size (2M,2M)
    """
    # kernel FB step size
    tau = coeff*1/(gamma*np.linalg.norm(ifft2(fft2(u)**2)) \
      + alpha*np.linalg.norm(ifft2(fft2(d_pad)**2)))
    # Step 1
    # calcul du gradient
    grad1 = gamma*convolve(u,convolve(u,K_pad)-g)# datafit term
    grad2 = alpha*convolve(d_pad,convolve(d_pad,K_pad)) # regularisation
    grad  = grad1 + grad2
    Kk    = K_pad - tau*grad # descente de gradient
    # Step 2
    # projection
    if simplex:
        Kk = Simplex(Kk)
    #
    return Kk

# One forward-bakward step for the image
def FBS_im(u,K_pad,vx,vy,g,mu,gamma=1,coeff=1):
    """
    one step of forward-backward algorithm for the image varaible
    Parameters
    ----------
        u       (np.array) : initial image variable of size (Nx,Ny)
        K_pad   (np.array) : padded kernel variable of size (Nx,Ny)
        vx,vy   (np.array) : dual variable of size (Nx,Ny)^2
        g       (np.array) : noisy blurred image of size (Nx,Ny)
        mu         (float) : regularization parameter for Total Variation (TV)
        gamma      (float) : weight on the datafit term
        coeff      (float) : coefficient on the gradient step size
        
    Returns
    -------
        uk     (np.array) : image variable after one FB step, of size (Nx,Ny)
    """
    # image FB step size
    tau = coeff*1/(np.sqrt(8)+gamma*np.linalg.norm(ifft2(fft2(K_pad)**2)))
    # Step 1
    # compute gradient
    nablap = divh(vx,vy) 
    nablax = gamma*convolve(K_pad,convolve(K_pad,u)-g)
    nablaF = nablap+nablax
    uk     = u - tau*nablaF
    # Step 2
    # projection on [0,1]
    uk[uk<0] = 0
    uk[uk>1] = 1
    return uk

# One forward-bakward step for v
def FBS_dual(u,vx,vy,mu,gamma=1,coeff=1):
    """
    One step of forward-backward algorithm for the auxiliary varaible
    Parameters
    ----------
        u       (np.array) : image variable (with relaxation) of size (Nx,Ny)
        vx,vy   (np.array) : initial dual variable of size (Nx,Ny)^2
        mu         (float) : regularization parameter for Total Variation (TV)
        gamma      (float) : weight on the datafit term
        coeff      (float) : coefficient on the gradient step size
    Returns
    -------
        vkx,vky (np.array) : dual variable after one FB step, of size (Nx,Ny)^2
    """
    # image FB step size
    tau = coeff*1/np.sqrt(8)
    # Gradient step p
    dxu,dyu  = nablah(u) 
    vkx      = vx + tau*dxu
    vky      = vy + tau*dyu 
    # Projection on B(mu)
    vkx, vky = projB(vkx,vky,mu)
    return vkx, vky


# Total energy
def Energy(u,K_pad,vx,vy,g,d_pad,alpha,mu,gamma=1):
    """
    Energy of function, for primal and dual varaible.
    Parameters
    ----------
        u       (np.array) : image variable of size (Nx,Ny)
        K_pad   (np.array) : padded kernel variable of size (Nx,Ny)
        vx,vy   (np.array) : dual variable of size (Nx,Ny)^2
        g       (np.array) : noisy blurred image of size (Nx,Ny)
        d_pad   (np.array) : derivarion for regularization
        alpha      (float) : parameter for kernel of regularization of Tikhonov type 
        mu         (float) : regularization parameter for Total Variation (TV)
        gamma      (float) : weight on the datafit term
    Returns
    -------
        Ep_K      (float) : primal energy of kernel
        Ep_x      (float) : primal energy of image
        Ed        (float) : dual energy of auxiliary variable
    """
    #
    conv1 = convolve(K_pad,u)
    conv2 = convolve(d_pad,K_pad)
    ux,uy = nablah(u)
    normu = np.abs(ux)+np.abs(uy)
    #
    Ep    = 0.5*gamma*np.linalg.norm(conv1-g)**2 \
          + 0.5*alpha*np.linalg.norm(conv2)**2\
          + mu*np.sum(normu)
    #
    Ed    = -np.sum(ux*vx)-np.sum(uy*vy)
    return Ep,Ed

# Total Gradient
def Gradient(u,K_pad,vx,vy,g,d_pad,alpha,mu,gamma=1):
    """
    Gradient according to kernel K and image x.
    Parameters
    ----------
        u       (np.array) : image variable of size (Nx,Ny)
        K_pad   (np.array) : padded kernel variable of size (Nx,Ny)
        vx,vy   (np.array) : dual variable of size (Nx,Ny)^2
        g       (np.array) : noisy blurred image of size (Nx,Ny)
        d_pad   (np.array) : derivarion for regularization
        alpha      (float) : parameter for kernel of regularization of Tikhonov type 
        mu         (float) : regularization parameter for Total Variation (TV)
        gamma      (float) : weight on the datafit term
    Returns
    -------
        grad_K     (float): norm of gradient for kernel
        grad_x     (float): norm of gradient for image
    """
    # gradient for Kernel
    grad1 = gamma*convolve(u,convolve(u,K_pad)-g)# datafit term
    grad2 = alpha*convolve(d_pad,convolve(d_pad,K_pad)) # regularisation
    grad  = grad1 + grad2
    #
    grad_K = np.linalg.norm(grad)
    #
    # gradient for image
    nablap = divh(vx,vy) 
    nablax = gamma*convolve(K_pad,convolve(K_pad,u)-g)
    nablaF = nablap+nablax
    #
    grad_x = np.linalg.norm(nablaF)
    #
    return grad_K, grad_x
    
        