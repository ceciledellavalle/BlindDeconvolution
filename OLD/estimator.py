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
from Codes.simplex import Simplex
import sys
import matplotlib.pyplot as plt


# Estimation of the Kernel
def Estimator_Lap(M,x_init,x_blurred,alpha,niter=100,simplex=False,nesterov=False):
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
    dx, dy = np.meshgrid(np.linspace(-1,1,2*M), np.linspace(-1,1,2*M))
    t      = np.sqrt(dx*dx+dy*dy)
    min_x = Nx//2+1-M-2
    max_x = Nx//2+M-1
    min_y = Ny//2+1-M-2
    max_y = Ny//2+M-1
    # Initialisation
    x_k    = np.zeros((Nx,Ny))
    #
    Jalpha = np.zeros(niter)
    ngrad  = np.zeros(niter)
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.zeros((Nx,Ny))
    d_pad[Nx//2-1:Nx//2+2,Ny//2-1:Ny//2+2] = d
    # Perform fft
    fft_xk = fft2(x_k)
    fft_xi = fft2(x_init)
    fft_xb = fft2(fftshift(x_blurred))
    fft_d  = fft2(d_pad)
    # compute gradient step as 2/L with L the lipschitz constant
    grad   = np.conjugate(fft_xi)*fft_xi\
            + alpha*np.conjugate(fft_d)*fft_d
    grad   = np.real(ifft2(grad))
    Lip    = np.linalg.norm(grad,ord=2)
    tau    = 2/Lip
    print("Pas constant de descente = ",tau)
    # Nesterov acceleration
    if nesterov:
        tk     = tau
        tkold  = tk
        x_old  = x_k.copy()
    for i in range(niter):
        # Step 1
        # calcul du gradient
        grad1 = np.conjugate(fft_xi)*(fft_xi*fft_xk-fft_xb)# datafit term
        grad2 = alpha*np.conjugate(fft_d)*fft_d*fft_xk # regularisation
        grad  = grad1 + grad2
        #
        # Step 3 
        # retour dans le domaine spatial
        cc       = np.real(ifft2(grad))
        ngrad[i] = np.linalg.norm(cc)
        # Step 2
        # descente de gradient
        x_k = x_k - tau*cc
        #
        # Step 4
        # annulation en dehors du support 
        # et projection sur le simplexe
        if simplex:
            proj                          = np.zeros((Nx,Ny))
            proj[min_x:max_x,min_y:max_y] = Simplex(x_k[min_x:max_x,min_y:max_y])
            x_k                           = proj.copy()
        # Step 5
        # Nesterov acceleration
        if nesterov:
            tk    = (1+math.sqrt(4*tkold+1))/2
            relax = 1+(tkold-1)/tk
            tkold = tk
            x_k   = relax*x_k + (1-relax)*x_old
            x_old = x_k.copy()
        # Step 6
        fft_xk = fft2(x_k)
        #
        # Step 7
        # Functional computation  with Parceval formula
        conv1 = np.real(ifft2(fft_xi*fft_xk))
        conv1 = fftshift(conv1)
        conv2 = np.real(ifft2(fft_d*fft_xk))
        conv2 = fftshift(conv2)
        Jalpha[i] = 0.5*np.linalg.norm(conv1-x_blurred)**2 \
             + 0.5*alpha*np.linalg.norm(conv2)**2
    # # plot
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5,3))
    # ax0.loglog(Jalpha)
    # ax0.set_title("Fonction J à minimiser")
    # ax1.plot(ngrad)
    # plt.show()
    return x_k[min_x:max_x,min_y:max_y], Jalpha


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
