"""
Image reconstruction function Function
---------
    Estimator_TV  : algorithm to compute reconstruct image under constraint
@author: Cecile Della Valle
@date: 15/12/2020
"""

# IMPORTATION
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import math

# Image reconstruction
def Estimator_TV(x_blurred,K,mu,niter=100):
    #Local parameters and matrix sizes
    M,_    = K.shape
    M      = M//2
    Nx, Ny = x_blurred.shape
    min_x  = Nx//2+1-M-2
    max_x  = Nx//2+M-1
    min_y  = Ny//2+1-M-2
    max_y  = Ny//2+M-1
    # Energy
    En     = np.zeros(niter)
    # Kernel padding, shift, fft
    K_padd = np.zeros((Nx,Ny))
    K_padd[min_x:max_x,min_y:max_y] = K
    fftK = fft2(fftshift(K_padd))
    # Image fft
    fftx = fft2(x_blurred)
    #
    # Primal dual algorithm
    # Parameters
    tau = 1/(math.sqrt(8)+1)# manually computed
    # initialisation
    v_init = np.zeros((Nx,Ny))
    v_bar  = v_init.copy()
    v_old  = v_init.copy() 
    v      = v_init.copy()
    px     = np.zeros((Nx,Ny)) 
    py     = np.zeros((Nx,Ny))
    #
    # local function projection on B(alpha)
    def projB(px,py,mu) :
        n,m = px.shape
        l,p = py.shape
        norm     = np.sqrt(px**2+py**2)
        cond     = norm>mu
        ppx, ppy = np.zeros((n,m)), np.zeros((l,p))
        ppx[cond] = mu*px[cond]/norm[cond]
        ppy[cond] = mu*py[cond]/norm[cond]
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
        # Projection on B(mu)
        px, py = projB(px,py,mu)
        #
        # Gradient step v
        nablap = divh(px,py) 
        fftv = fft2(v) 
        fftnx = np.conjugate(fftK)*(fftK*fftv-fftx) 
        nablax = np.real(ifft2(fftnx)) 
        nablaF = nablap+nablax
        # projection on [0,1]
        v -= tau*nablaF
        v[v<0] = 0
        v[v>1] = 1
        # relaxation
        v_bar =2*v -v_old
        v_old = v
        #
        # Energy  with Parceval formula
        conv1 = np.real(ifft2(fftK*fftv))
        conv1 = fftshift(conv1)
        norm1 = np.abs(divh(v_bar,v_bar))
        En[i] = 0.5*np.linalg.norm(conv1-x_blurred)**2 \
                + mu*np.sum(norm1)
    return v_bar, En