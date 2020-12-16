"""
Kernel estimation Function
---------
    Estimator     : algorithm to compute an estimation of a Kernel
@author: Cecile Della Valle
@date: 09/11/2020
"""

# IMPORTATION
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import math
from simplex import Simplex
import sys


# Estimation of the Kernel
def Estimator_Lap(M,x_init,x_blurred,alpha,mu,tau=100,niter=100):
    """
    Estimation d'un noyau de convolution par méthode variationnelle
    Régularisation par convolution avec Laplacien
    Fista Algorithm
    Parameters
    ----------
        M (int)                : size parameter of 2D-Kernel (2M)^2
        x_init (numpy array)   : image size c x Nx x Ny
        x_blurred (numpy array): blurred image c x Nx x Ny
        alpha (float)          : regularisation parameter
        niter (int)            : number of iterations
    Returns
    -------
        K (numpy array): the estimated Kernel
    """
    # Local parameters
    Nx, Ny = x_init.shape
    min_x = Nx//2+1-M-2
    max_x = Nx//2+M-1
    min_y = Ny//2+1-M-2
    max_y = Ny//2+M-1
    # Initialisation
    x_k   = np.random.randn(Nx,Ny)
    x_old = x_k.copy() 
    tkold = 1 
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.zeros((Nx,Ny))
    d_pad[Nx//2:Nx//2+3,Ny//2:Ny//2+3] = d
    # Perform fft
    fft_xi = fft2(x_init)
    fft_xb = fft2(x_blurred)
    fft_d  = fft2(d_pad)
    for _ in range(niter):
        # Step 1
        # calcul du gradient
        grad1 = 2*mu*np.conjugate(fft_xi)*(fft_xi*x_k-fft_xb)# datafit term
        grad2 = 2*alpha*np.conjugate(fft_d)*fft_d*x_k # regularisation
        grad = grad1 + grad2
        grad = grad/(Nx*Ny)
        #
        x_k = x_k - tau*grad
        #
        # Step 2 
        # annulation en dehors du support 
        # et projection sur le simplexe
        # x_k                          = np.zeros((Nx,Ny)) 
        # x_k[min_x:max_x,min_y:max_y] = Simplex(y_k[min_x:max_x,min_y:max_y])
        # Nesterov acceleration
        tk = (1+math.sqrt(4*tkold+1))/2 
        relax = 1+(tkold-1)/tk 
        tkold = tk 
        x_k = relax*x_k + (1-relax)*x_old 
        x_old = x_k.copy()
        #
    # Kernel K reconstruction
    x_k = np.real(ifft2(x_k))
    x_k = fftshift(x_k)
    K = x_k[min_x:max_x,min_y:max_y]
    #
    return K


# Estimation of the Kernel
def Estimator(M,x_init,x_blurred,alpha,mu,niter=100):
    """
    Estimation d'un noyau de convolution par méthode variationnelle
    Fista Algorithm
    Parameters
    ----------
        M (int)                : size parameter of 2D-Kernel (2M)^2
        x_init (numpy array)   : image size c x Nx x Ny
        x_blurred (numpy array): blurred image c x Nx x Ny
        alpha (float)          : regularisation parameter
        niter (int)            : number of iterations
    Returns
    -------
        K (numpy array): the estimated Kernel
    """
    # Local parameters
    Nx, Ny = x_init.shape
    min_x = Nx//2+1-M-2
    max_x = Nx//2+M-1
    min_y = Ny//2+1-M-2
    max_y = Ny//2+M-1
    # Initialisation
    p2_bar = np.random.randn(Nx,Ny)
    p2_old = p2_bar 
    tkold = 1 
    # FFT images
    fft_u = fft2(x_init)
    fft_g = fft2(x_blurred)
    # Rescale alpha
    if alpha>1 :
        alpha = Nx*Ny*alpha
    #
    for _ in range(niter):
        # Actualisation p1
        # calcul de prox f dans le domaine de Fourier
        hat_p2_bar = fft2(-p2_bar/2/alpha) 
        hat_prox = (2*alpha*hat_p2_bar \
            + mu*np.conjugate(fft_u)*fft_g)\
             /(2*alpha \
             + mu*np.conjugate(fft_u)*fft_u) 
        # retour dans le domaine spatial
        prox = np.real(ifft2(hat_prox)) 
        #
        p1 = - p2_bar - 2*alpha*prox 
        #
        # Actualisation of p2
        # annulation en dehors du support 
        # et projection sur le simplexe
        p1_shift = fftshift(p1) 
        proj = np.zeros((Nx,Ny)) 
        proj[min_x:max_x,min_y:max_y] = Simplex(-p1_shift[min_x:max_x,min_y:max_y]/2/alpha)
        proj = fftshift(proj) 
        #
        p2 = - p1 - 2*alpha*proj 
        # Relaxation (FISTA)
        tk = (1+math.sqrt(4*tkold+1))/2 
        relax = 1+(tkold-1)/tk 
        tkold = tk 
        p2_bar = relax*p2 + (1-relax)*p2_old 
        p2_old = p2 
        #
    # Kernel K reconstruction
    K = fftshift(-(p1 + p2)/2/alpha) 
    K = K[min_x:max_x,min_y:max_y]
    #
    return K

#======================================================================
#======================================================================
#
# Estimation of the Kernel
# with smoothing regularisation
def Estimator_Lap_temp(M,x_init,x_blurred,alpha,mu,tau=100,niter=100):
    """
    Estimation d'un noyau de convolution par méthode variationnelle
    Régularisation par convolution avec Laplacien
    Fista Algorithm
    Parameters
    ----------
        M (int)                : size parameter of 2D-Kernel (2M)^2
        x_init (numpy array)   : image size c x Nx x Ny
        x_blurred (numpy array): blurred image c x Nx x Ny
        alpha (float)          : regularisation parameter
        niter (int)            : number of iterations
    Returns
    -------
        K (numpy array): the estimated Kernel
    """
    # Local parameters
    Nx, Ny = x_init.shape
    min_x = Nx//2+1-M-2
    max_x = Nx//2+M-1
    min_y = Ny//2+1-M-2
    max_y = Ny//2+M-1
    # Initialisation
    x_k   = np.random.randn(Nx,Ny)
    x_old = x_k.copy() 
    tkold = 1 
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.zeros((Nx,Ny))
    d_pad[Nx//2:Nx//2+3,Ny//2:Ny//2+3] = d
    # Rescale alpha
    if alpha>1 :
        alpha = Nx*Ny*alpha
    #
    def myconvolv(a,b):
        # a and b must be of same size !
        m,n   = a.shape
        # Perform fft
        fft_a = fft2(a)
        fft_b = fft2(b) 
        # Compute multiplication and inverse fft
        cc    = np.real(ifft2(fft_a*fft_b))
        cc    = np.roll(cc, -int(m//2+1),axis=0)
        cc    = np.roll(cc, -int(n//2+1),axis=1)
        return cc
    #
    def myconvolvT(a,b):
        # a and b must be of same size !
        m,n   = a.shape
        # Perform fft
        fft_a = fft2(a)
        fft_b = fft2(b) 
        # Compute conjugaison, 
        # multiplication and inverse fft
        cc    = np.real(ifft2(np.conjugate(fft_a)*fft_b))
        cc    = np.roll(cc, -int(m//2+1),axis=0)
        cc    = np.roll(cc, -int(n//2+1),axis=1)
        return cc
    #
    for _ in range(niter):
        # Step 1
        # calcul du gradient
        conv1 = myconvolvT(x_k,x_init)
        grad1 = 2*mu*myconvolv(x_init,conv1-x_blurred) # datafit term
        conv2 = myconvolv(d_pad,x_k)
        grad2 = 2*alpha*myconvolvT(d_pad,conv2) # regularisation
        grad = grad1 + grad2
        grad = grad/(Nx*Ny)
        #
        x_k = x_k - tau*grad
        #
        # Step 2 
        # annulation en dehors du support 
        # et projection sur le simplexe
        # x_k                          = np.zeros((Nx,Ny)) 
        # x_k[min_x:max_x,min_y:max_y] = Simplex(y_k[min_x:max_x,min_y:max_y])
        # Nesterov acceleration
        tk = (1+math.sqrt(4*tkold+1))/2 
        relax = 1+(tkold-1)/tk 
        tkold = tk 
        x_k = relax*x_k + (1-relax)*x_old 
        x_old = x_k.copy()
        #
    # Kernel K reconstruction
    K = x_k[min_x:max_x,min_y:max_y]
    #
    return K