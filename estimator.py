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

# FUNCTION DEFINITION
def Estimator(M,x_init,x_blurred,alpha,mu,niter = 1000):
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
    support_x = np.arange(Nx/2+1-M,Nx/2+M)
    support_y = np.arange(Ny/2+1-M,Ny/2+M)
    # Initialisation
    p2_bar = np.random.randn(Nx,Ny)
    p2_old = p2_bar 
    tkold = 1 
    # FFT images
    fft_u = fft2(x_init)
    fft_g = fft2(x_blurred)
    #
    for _ in range(niter):
        # Actualisation p1
        # calcul de prox f dans le domaine de Fourier
        hat_p2_bar = fft2(-p2_bar/2/alpha) 
        hat_prox = (2*alpha*hat_p2_bar \
            + mu*np.conjugate(fft_u).dot(fft_g))\
             /(2*alpha \
             + mu*np.conjugate(fft_u).dot(fft_u)) 
        # retour dans le domaine spatial
        prox = np.real(ifft2(hat_prox)) 
        #
        p1 = - p2_bar - 2*alpha*prox 
        #
        # Actualisation of p2
        # annulation en dehors du support 
        # et projection sur le simplexe
        p1_shift = fftshift(p1) 
        proj = np.zeros(Nx,Ny) 
        proj[support_x,support_y] = Simplex(-p1_shift[support_x,support_y]/2/alpha)
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
    K = K[support_x,support_y]
    #
    return K