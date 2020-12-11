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

# Image reconstruction
def Estimator_TV(x_blurred,K,alpha,mu,niter=100):
    #Local parameters and matrix sizes
    M,_    = K.shape
    M      = M//2
    Nx, Ny = x_blurred.shape
    min_x  = Nx//2+1-M-2
    max_x  = Nx//2+M-1
    min_y  = Ny//2+1-M-2
    max_y  = Ny//2+M-1
    # Kernel padding, shift, fft
    K_padd = np.zeros((Nx,Ny))
    K_padd[min_x:max_x,min_y:max_y] = K
    fftK = fft2(fftshift(K_padd))
    # Image fft
    fftx = fft2(x_blurred)
    #
    # Primal dual algorithm
    # Parameters
    tau = 1/(math.sqrt(8)+mu)# manually computed
    # initialisation
    v_init = np.zeros((Nx,Ny))
    v_bar  = v_init.copy()
    v_old  = v_init.copy() 
    v      = v_init.copy()
    px     = np.zeros((Nx,Ny)) 
    py     = np.zeros((Nx,Ny))
    #
    # local function projection on B(alpha)
    def projB(px,py,alpha) :
        n,m = px.shape
        l,p = py.shape
        norm     = np.sqrt(px**2+py**2)
        cond     = norm>alpha
        ppx, ppy = np.zeros((n,m)), np.zeros((l,p))
        ppx[cond] = alpha*px[cond]/norm[cond]
        ppy[cond] = alpha*py[cond]/norm[cond]
        return ppx, ppy
    # local divh
    def divh(px,py):
        dd = np.zeros((Nx,Ny))
        dd[1:,:] += px[1:,:]-px[:-1,:]
        dd[:,1:] += py[:,1:]-py[:,:-1]
        return dd
    # local nablah
    def nablah(v):
        ddx, ddy = np.zeros((Nx,Ny)), np.zeros((Nx,Ny))
        ddx[:-1] = v[1:,:]-v[:-1,:]
        ddy[:-1] = v[1:,:]-v[:-1,:]
        return ddx, ddy
    #
    for i in range(niter):
        #
        # Gradient step p
        dxu,dyu = nablah(v_bar) 
        px = px + tau*dxu
        py = py + tau*dyu 
        # Projection on B(\alpha)
        px, py = projB(px,py,alpha)
        #
        # Gradient step v
        nablap = divh(px,py) 
        fftv = fft2(v) 
        fftnx = mu*np.conjugate(fftK)*(fftK*fftv-fftx) 
        nablax = np.real(ifft2(fftnx)) 
        nablaF = nablap+nablax
        # projection on [0,1]
        v -= tau*nablaF
        v[v<0] = 0
        # v[v>1] = 1
        # relaxation
        v_bar =2*v -v_old
        v_old = v

    return v_bar

# Estimation of the Kernel
def Estimator_Lap(M,x_init,x_blurred,alpha,mu,niter=100):
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
    # Gradient descent parameter
    tau = 100
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
    for _ in range(niter):
        # Step 1
        # calcul du gradient
        conv1 = myconvolv(x_k,x_init)
        grad1 = 2*mu*(myconvolv(x_init.T,conv1)-myconvolv(x_init.T,x_blurred)) # datafit term
        conv2 = myconvolv(d_pad,x_k)
        grad2 = 2*alpha*myconvolv(d_pad,conv2) # regularisation
        grad = grad1 + grad2
        #
        y_k = x_k - tau*grad
        #
        # Step 2 
        # annulation en dehors du support 
        # et projection sur le simplexe
        x_k                          = np.zeros((Nx,Ny)) 
        x_k[min_x:max_x,min_y:max_y] = Simplex(y_k[min_x:max_x,min_y:max_y])
        # Relaxation (FISTA)
        tk = (1+math.sqrt(4*tkold+1))/2 
        relax = 1+(tkold-1)/tk 
        tkold = tk 
        x_k = relax*x_k + (1-relax)*x_old 
        x_old = x_k.copy()
        #
    # Kernel K reconstruction
    K = x[min_x:max_x,min_y:max_y]
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
