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



def violetBD(K_in,x_in,x_blurred,\
                  alpha,mu,theta=1,\
                  niter=200,\
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
           proj_simplex  (bool) : projection of Kernel over simplex space
       Returns
       ----------
           K         (np.array) : estimation of Kernel size (2M,2M)
           x         (np.array) : denoised deconvolved image size (Nx,Ny)
           Ep        (np.array) : Primal energy size (niter)
           Ed        (np.array) : Dual energy size (niter) 
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
    Ep     = np.zeros(niter) # primal energy
    Ed     = np.zeros(niter) # dual energy
    Ki     = K_in.copy() # kernel
    xi     = x_in.copy() # image
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
    # gradient step
    taux = 10**-4 # image FB step size
    tauK = 10**-8 # kernel FB step size
    taup = 10**-4 # dual function FB step size
    #
    for i in range(niter):
        # FBS for image
        xi = FBS_im(xi,Ki,px,py,x_blurred,regx,taux,param)
        # FBS for kernel
        Ki = FBS_ker(xi,Ki,x_blurred,regK,tauK,d_pad,param,simplex=proj_simplex)
        # primal energy
        Ep[i],_ = Energy(xi,Ki,px,py,x_blurred,regK,regx,d_pad,param)
        # test
        if (i>1):
            if (Ep[i] > Ep[i-1]):
                print("stops prematurely at {} iterations : the primal energy rises".format(i))
                return Ki,xi,Ep,Ed
        # FBS for v (dual of TV)
        px,py = FBS_dual(xi,px,py,mu,taup)
        # dual energy
        _,Ed[i] = Energy(xi,Ki,px,py,x_blurred,regK,regx,d_pad,param)
        # relaxation
        xtemp = xi.copy()
        xi     = xi + theta*(xi-xold)
        xold   = xtemp.copy()
    return Ki,xi,Ep,Ed
 
# One forward-bakward step for the image
def FBS_im(u,K,vx,vy,g,mu,tau,dict_param):
    """
    one step of forward-backward algorithm for the image varaible
    Parameters
    ----------
        u       (np.array) : initial image variable of size (Nx,Ny)
        K       (np.array) : kernel variable of size (2M,2M)
        vx,vy   (np.array) : dual variable of size (Nx,Ny)^2
        g       (np.array) : noisy blurred image of size (Nx,Ny)
        mu         (float) : regularization parameter for Total Variation (TV)
        tau        (float) : gradient step size
        dict_param (tuple) : size of kernel, image and padding parameters
        
    Returns
    -------
        uk     (np.array) : image variable after one FB step, of size (Nx,Ny)
    """
    # local parameters and matrix sizes
    M,Nx,Ny,min_x,max_x,min_y,max_y = dict_param
    # Kernel padding
    K_pad = np.zeros((Nx,Ny))
    K_pad[min_x:max_x,min_y:max_y] = K
    # Step 1
    # compute gradient
    nablap = divh(vx,vy) 
    nablax = convolve(K_pad,convolve(K_pad,u)-g)
    nablaF = nablap+nablax
    uk     = u - tau*nablaF
    # Step 2
    # projection on [0,1]
    uk[uk<0] = 0
    uk[uk>1] = 1
    return uk

# One forward-bakward step for the kernel
def FBS_ker(u,K,g,alpha,tau,d_pad,dict_param,simplex=True):
    """
    one step of forward-backward algorithm for the image varaible
    Parameters
    ----------
        u       (np.array) : image variable of size (Nx,Ny)
        K       (np.array) : initial kernel variable of size (2M,2M)
        g       (np.array) : noisy blurred image of size (Nx,Ny)
        alpha      (float) : parameter for kernel of regularization of Tikhonov type 
        tau        (float) : gradient step size
        d_pad   (np.array) : derivarion for regularization
        dict_param (tuple) : size of kernel, image and padding parameters
        simplex     (bool) : boolean, True if projection on simplex applied for kernel
    Returns
    -------
        Kk     (np.array) : kernel variable after one FB step, of size (2M,2M)
    """
    # local parameters and matrix sizes
    M,Nx,Ny,min_x,max_x,min_y,max_y = dict_param
    # Initialisation
    Kk                          = np.zeros((Nx,Ny))
    Kk[min_x:max_x,min_y:max_y] = K
    # Step 1
    # calcul du gradient
    grad1 = convolve(u,convolve(u,Kk)-g)# datafit term
    grad2 = alpha*convolve(d_pad,convolve(d_pad,Kk)) # regularisation
    grad  = grad1 + grad2
    Kk = Kk - tau*grad # descente de gradient
    # Step 2
    # projection
    if simplex:
        proj                          = np.zeros((Nx,Ny))
        proj[min_x:max_x,min_y:max_y] = Simplex(Kk[min_x:max_x,min_y:max_y])
        Kk                            = proj.copy()
    #
    return Kk[min_x:max_x,min_y:max_y]

# One forward-bakward step for v
def FBS_dual(u,vx,vy,mu,tau):
    """
    one step of forward-backward algorithm for the image varaible
    Parameters
    ----------
        u       (np.array) : image variable of size (Nx,Ny)
        vx,vy   (np.array) : initial dual variable of size (Nx,Ny)^2
        mu         (float) : regularization parameter for Total Variation (TV)
        tau        (float) : gradient step size
    Returns
    -------
        vkx,vky (np.array) : dual variable after one FB step, of size (Nx,Ny)^2
    """
    # Gradient step p
    dxu,dyu  = nablah(u) 
    vkx      = vx + tau*dxu
    vky      = vy + tau*dyu 
    # Projection on B(mu)
    vkx, vky = projB(vkx,vky,mu)
    return vkx, vky


# Total energy
def Energy(u,K,vx,vy,g,alpha,mu,d_pad,dict_param):
    """
    Energy of function, for primal and dual varaible.
    Parameters
    ----------
        u       (np.array) : image variable of size (Nx,Ny)
        K       (np.array) :  kernel variable of size (2M,2M)
        vx,vy   (np.array) : dual variable of size (Nx,Ny)^2
        g       (np.array) : noisy blurred image of size (Nx,Ny)
        alpha      (float) : parameter for kernel of regularization of Tikhonov type 
        mu         (float) : regularization parameter for Total Variation (TV)
        d_pad   (np.array) : derivarion for regularization
        dict_param (tuple) : size of kernel, image and padding parameters
    Returns
    -------
        Ep     (np.array) : primal energy of size (niter)
        Ed     (np.array) : dual energy of size (niter)
    """
    # local parameters and matrix sizes
    M,Nx,Ny,min_x,max_x,min_y,max_y = dict_param
    # Kernel padding
    K_pad = np.zeros((Nx,Ny))
    K_pad[min_x:max_x,min_y:max_y] = K
    #
    conv1 = convolve(K_pad,u)
    conv2 = convolve(d_pad,K_pad)
    ux,uy = nablah(u)
    normu = np.abs(ux)+np.abs(uy)
    #
    Ep    = 0.5*np.linalg.norm(conv1-g)**2 \
          + 0.5*alpha*np.linalg.norm(conv2)**2\
          + mu*np.sum(normu)
    #
    Ed    = -np.sum(ux*vx)-np.sum(uy*vy)
    return Ep,Ed


        