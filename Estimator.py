"""
Kernel estimation Function
---------
    Estimator     : algorithm to compute an estimation of a Kernel
@author: Cecile Della Valle
@date: 09/11/2020
"""

# IMPORTATION
import numpy as np
import sys

# FUNCTION DEFINITION
def Estimator(M,x_init,x_blurred,alpha,mu,niter = 1000):
    """
    Estimation d'un noyau de convolution par méthode variationnelle
    Fista Algorithm
    Parameters
    ----------
        M (int)
        x_init (numpy array)   : image size c x Nx x Ny
        x_blurred (numpy array): blurred image c x Nx x Ny
        alpha (float)          : regularisation parameter
        niter (int)            : number of iterations
    Returns
    -------
        K (numpy array): the estimated Kernel
    """
    # Local parameters
    c, Nx,Ny = x_init.shape
    support x = np.arange(Nx/2+1−M,Nx/2+M)
    support y = np.arange(Ny/2+1−M,Ny/2+M)
    # Initialisation
    p2 bar = randn(Nx,Ny) ; p2 old = p2 bar 
    tkold = 1 
    # FFT images
    # Descente de gradient
    #
    for n=1:niter
    # Actualisation p1
    # calcul de prox f dans le domaine de Fourier
    hat p2 bar = fft2(−p2 bar/2/alpha) 
    hat prox = (2∗alpha∗hat p2 bar + mu∗conj(fft u).∗fft g)\
        /(2∗alpha + mu∗real(conj(fft u).∗fft u)) 
    # retour dans le domaine spatial
    prox = real(ifft2(hat prox)) 
    #
    p1 = − p2 bar − 2∗alpha∗prox 
    #
    # Actualisation of p2
    # annulation en dehors du support et projection sur le simplexe
    p1 shift = fftshift(p1) 
    proj = zeros(Nx,Ny) 
    proj(support x,support y) = Simplexe(−p1 shift(support x,support y)/2/lambda) ;
    proj = fftshift(proj) 
    #
    p2 = − p1 − 2∗alpha∗proj 
    # Relaxation (FISTA)
    tk = (1+sqrt(4∗tkold+1))/2 
    relax = 1+(tkold−1)/tk 
    tkold = tk 
    p2_bar = relax∗p2 + (1−relax)∗p2_old 
    p2_old = p2 
    #
    # Kernel K reconstruction
    K = fftshift(−(p1 + p2)/2/alpha) 
    K = K(support x,support y)
    #
    return K