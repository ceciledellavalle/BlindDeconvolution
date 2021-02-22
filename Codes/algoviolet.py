"""
Kernel estimation Function
---------
      violet  : algorithm to compute blind deconvolution by minimization method
            of one FBS step convexe, one FBS step convexe, one FBS concave
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


def violetBD(K_in,x_in,x_blurred,alpha,mu,\
                  alte=10,niter=200,\
                  proj_simplex=False):
    """
    one FBS step convexe, one FBS step convexe, one FBS concave
    extended chambolle-pock
    """
    # initialisation
    Etot     = np.zeros(alte*niter_Lap+alte*niter_TV)
    Ki       = K_in.copy()
    xi       = x_in.copy()
    # local parameters and matrix sizes
    M,_    = K_in.shape
    M      = M//2
    Nx, Ny = x_blurred.shape
    min_x  = Nx//2+1-M-2
    max_x  = Nx//2+M-1
    min_y  = Ny//2+1-M-2
    max_y  = Ny//2+M-1
    # Derivation
    d      = -np.ones((3,3))
    d[1,1] = 8
    d_pad  = np.zeros((Nx,Ny))
    d_pad[Nx//2-1:Nx//2+2,Ny//2-1:Ny//2+2] = d
    #
    for i in ramge(niter):
    # FBS for image
    # FBS for kernel
    # FBS for v (dual of TV)
    return Ki,xi,Etot
 
# One forward-bakward step for the image
def FBS_im(u,K,vx,vy,g,mu,tau,dict_param):
    # local parameters and matrix sizes
    M,Nx,Ny,min_x,max_x,min_y,max_y = dict_param
    # Kernel padding, shift, fft
    K_padd = np.zeros((Nx,Ny))
    K_padd[min_x:max_x,min_y:max_y] = K
    fftK = fft2(fftshift(K_padd))
    # Image fft
    fftx = fft2(g)
    # Step 1
    # compute gradient
    nablap = divh(vx,vy) 
    fftu   = fft2(u) 
    nablax = np.real(ifft2(np.conjugate(fftK)*(fftK*fftu-fftx) )) 
    nablaF = nablap+nablax
    uk     = u - tau*nablaF
    # Step 2
    # projection on [0,1]
    uk[uk<0] = 0
    uk[uk>1] = 1
    return uk

# One forward-bakward step for the kernel
def FBS_kernel(u,K,g,alpha,tau,d_pad,dict_param,simplex=True):
    # local parameters and matrix sizes
    M,Nx,Ny,min_x,max_x,min_y,max_y = dict_param
    # Initialisation
    Kk                          = np.zeros((Nx,Ny))
    Kk[min_x:max_x,min_y:max_y] = K
    # Perform fft
    fftK = fft2(Kk)
    fftu = fft2(u)
    fftg = fft2(fftshift(g))
    fft_d  = fft2(d_pad)
    # Step 1
    # calcul du gradient
    grad1 = np.conjugate(fft_xi)*(fft_xi*fft_xk-fft_xb)# datafit term
    grad2 = alpha*np.conjugate(fft_d)*fft_d*fft_xk # regularisation
    grad  = grad1 + grad2
    cc       = np.real(ifft2(grad)) # retour dans le domaine spatial
    x_k = x_k - tau*cc # descente de gradient
    # Step 2
    # projection
    if simplex:
        proj                          = np.zeros((Nx,Ny))
        proj[min_x:max_x,min_y:max_y] = Simplex(Kk[min_x:max_x,min_y:max_y])
        Kk                            = proj.copy()
    #
    return Kk[min_x:max_x,min_y:max_y]

# One forward-bakward step for v
def FBS_v(u,vx,vy):
    # Gradient step p
    dxu,dyu = nablah(u) 
    vx      = vx + tau*dxu
    vy      = vy + tau*dyu 
    # Projection on B(mu)
    vx, vy  = projB(px,py,mu)
    return vx, vy

# function projection on B(alpha)
def projB(px,py,mu) :
    n,m = px.shape
    l,p = py.shape
    norm     = np.sqrt(px**2+py**2)
    cond     = norm>mu
    ppx, ppy = np.zeros((n,m)), np.zeros((l,p))
    ppx[cond] = mu*px[cond]/norm[cond]
    ppy[cond] = mu*py[cond]/norm[cond]
    return ppx, ppy
# divh
def divh(px,py):
    dd = np.zeros((Nx,Ny))
    dd[1:,:] += px[1:,:]-px[:-1,:]
    dd[:,1:] += py[:,1:]-py[:,:-1]
    return dd
# nablah
def nablah(v):
    ddx, ddy = np.zeros((Nx,Ny)), np.zeros((Nx,Ny))
    ddx[:-1] = v[1:,:]-v[:-1,:]
    ddy[:-1] = v[1:,:]-v[:-1,:]
    return ddx, ddy 
# Total energy
def Energy(u,K,g,alpha,mu):
    # Perform fft
    fftK = fft2(K)
    fftu = fft2(u)
    fftg = fft2(fftshift(g))
    fft_d  = fft2(d_pad)
    # Functional computation  with Parceval formula
    conv1 = np.real(ifft2(fft_xi*fft_xk))
    conv1 = fftshift(conv1)
    conv2 = np.real(ifft2(fft_d*fft_xk))
    conv2 = fftshift(conv2)
    vx,vy = nablah(v_bar)
    normv = np.abs(vx)+np.abs(vy)
    E  = 0.5*np.linalg.norm(conv1-x_blurred)**2 \
         + 0.5*alpha*np.linalg.norm(conv2)**2\
         + mu*np.sum(normv)
    return E


        