"""
Kernel estimation Function
---------
    AlternatingBD   : algorithm to compute blind deconvolution by alternating minimization method
    Estimator_TV    : Chambolle-Pock algorithm for deblurring denoising
    Estimator_Lap   : FB algorithm with regularization of Tikhonov type to estimate a convolution kernel

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
from Codes.display import Display_ker
from Codes.display import Display_im


def AlternatingBD(K_in,x_in,x_blurred,alpha,mu,\
                  alte=10,niter_TV=200,niter_Lap =200,\
                  proj_simplex=False):
    """
    Alternating estimation of a blind deconvolution.
    Parameters
    ----------
        K_in      (numpy array) : initial kernel of size (2M,2M)
        x_in     (numpy array) : initial image of size (Nx,Ny)
        x_blurred (numpy array) : blurred and noisy image of size (Nx,Ny)
        alpha           (float) : regularisation parameter for Tikhonov type regularization
        mu              (float) : regularisation parameter for TV
        alte              (int) : number of alternating iterations
        niter_TV          (int) : number of iterations for image reconstruction
        niter_TV          (int) : number of iterations for kernel reconstruction
        proj_simplex     (bool) : boolean, True if projection of kernel on simplex space
    Returns
    --------
        Ki      (numpy array) : final estimated kernel of size (2M,2M)
        xi      (numpy array) : final deblurred denoised image of size (Nx,Ny)
        Etot    (numpy array) : primal energy through every FB step
    """
    # initialisation
    Etot     = np.zeros(alte*niter_Lap+alte*niter_TV)
    Ki       = K_in.copy()
    xi       = x_in.copy()
    _,_,resK = Estimator_Lap(Ki,xi,x_blurred,alpha,niter=1)
    # rescaling
    M        = Ki.shape[0]//2
    Nx,Ny    = xi.shape
    regK     = alpha*(2*M)**2
    regx     = mu/Nx/Ny
    for i in range(alte):
        # First estimation of image
        xold        = xi.copy()
        xi,E2,resx  = Estimator_TV(Ki,xi,x_blurred,regx,niter=niter_TV)
        # energy
        Etot[i*(niter_Lap+niter_TV):i*niter_Lap+(i+1)*niter_TV] = E2 +resK
        # Second estimation of Kernel
        Kold = Ki.copy()
        Ki,E1,resK = Estimator_Lap(Ki,xi,x_blurred,regK,niter=niter_Lap,\
                                    simplex=proj_simplex)
        # energy
        Etot[(i+1)*niter_TV+i*niter_Lap:(i+1)*(niter_Lap+niter_TV)] = E1 +resx
    #
    # display
    Display_ker(Ki,K_in)
    # display
    Display_im(xi,xold)
    return Ki,xi,Etot
        
# Image reconstruction
def Estimator_TV(K,x_zero,x_blurred,mu,niter=100):
    """
    De-blurring, de-noising Chambolle-Pock algorithm
    Regularization with Total Varaition TV
    Fista Algorithm
    Parameters
    ----------
        K         (numpy array) : kernel of size (2M,2M)
        x_zero    (numpy array) : initial image of size (Nx,Ny)
        x_blurred (numpy array) : blurred and noisy image of size (Nx,Ny)
        mu              (float) : regularisation parameter
        niter             (int) : number of iterations
    Returns
    -------
        v_bar      (numpy array): denoised deblurred image of size (Nx,Ny)
        En         (numpy array): total energy of the deconvolution denoising criterion
        res        (numpy array): TV of v_bar
    """
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
    tau = 0.01/(math.sqrt(8)+1)# manually computed
    # initialisation
    v_init = x_zero
    v_bar  = v_init.copy()
    v_old  = v_init.copy() 
    v      = v_init.copy()
    px     = np.zeros((Nx,Ny)) 
    py     = np.zeros((Nx,Ny))
    #
    # local function projection on B(alpha)
    def projB(px,py,mu) :
        norm     = np.sqrt(px**2+py**2)
        cond     = norm>mu
        ppx, ppy = px.copy(), py.copy()
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
        px      = px + tau*dxu
        py      = py + tau*dyu 
        # Projection on B(mu)
        px, py  = projB(px,py,mu)
        #
        # Gradient step v
        nablap = divh(px,py) 
        fftv   = fft2(v) 
        fftnx  = np.conjugate(fftK)*(fftK*fftv-fftx) 
        nablax = np.real(ifft2(fftnx)) 
        nablaF = nablap+nablax
        v     -= tau*nablaF
        # projection on [0,1]
        v[v<0] = 0
        v[v>1] = 1
        # relaxation
        v_bar =2*v -v_old
        v_old = v
        #
        # Energy  with Parceval formula
        conv1 = np.real(ifft2(fftK*fftv))
        vx,vy = nablah(v_bar)
        normv = np.abs(vx)+np.abs(vy)
        En[i] = 0.5*np.linalg.norm(conv1-x_blurred)**2 \
                + mu*np.sum(normv)
    res = mu*np.sum(normv)
    return v_bar, En, res


# Estimation of the Kernel
def Estimator_Lap(K_zero,x_init,x_blurred,alpha,niter=100,simplex=False,nesterov=False):
    """
    Estimation of a kernel of convolution, knowing blurred image and true image
    Regularization of Tikhonov type with derivative
    Fista Algorithm
    Parameters
    ----------
        K_zero    (numpy array) : initial kernel of size (2M,2M)
        x_init    (numpy array) : true image of size (Nx,Ny)
        x_blurred (numpy array) : blurred and noisy image f size (Nx,Ny)
        alpha           (float) : regularisation parameter
        niter             (int) : number of iterations
        simplex          (bool) : boolean, True if projection of kernel on simplex space
        nesterov         (bool) : boolean, True if nesterov acceleration
    Returns
    -------
        x_k          (numpy array): the estimated Kernel of size (2M,2M)
        Jalpha       (numpy array): energy of the deconvolution criterion
        res          (numpy array): regularization of Tikhonov type
    """
    # Local parameters
    M      = K_zero.shape[0]//2
    Nx, Ny = x_init.shape
    dx, dy = np.meshgrid(np.linspace(-1,1,2*M), np.linspace(-1,1,2*M))
    t      = np.sqrt(dx*dx+dy*dy)
    min_x = Nx//2+1-M-2
    max_x = Nx//2+M-1
    min_y = Ny//2+1-M-2
    max_y = Ny//2+M-1
    # Initialisation
    x_k                          = np.zeros((Nx,Ny))
    x_k[min_x:max_x,min_y:max_y] = K_zero
    #
    Jalpha = np.zeros(niter)
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
    tau    = 0.001*2/Lip
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
    res = 0.5*alpha*np.linalg.norm(conv2)**2 
    return x_k[min_x:max_x,min_y:max_y], Jalpha, res
    
