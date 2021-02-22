"""
Tools functions to load, blurr and show images or data
---------
    Display        : show initial image, blurred image, kernel
@author: Cecile Della Valle
@date: 09/11/2020
"""

# IMPORTATION
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from Codes.dataprocess import Blurr


def Display(K_init, K_alpha, Ninterp = 100):
    """
    Compare initial Kernel and reconstruct Kernel with regularization parameter alpha.
    Kernel diagonal are interpolatre on finer grid.

    Parameters
    ----------
        K_init    (numpy array) : initial kernel (2Mx2M size)
        K_alpha   (numpy array) : reconstruct kernel (2Mx2M size)
        Ninterp         (float) : size of grid of interpolation
    Returns
    ----------
        -
    """
    # Interpolate the Kernel diagonal
    Nx     = np.linspace(-1,1,K_init.shape[0])
    fi     = interpolate.interp1d(Nx, np.diag(K_init))
    fa     = interpolate.interp1d(Nx, np.diag(K_alpha))
    Nxnew  = np.linspace(-1,1, Ninterp)
    K_i    = fi(Nxnew)
    K_a    = fa(Nxnew)
    # define graph
    fig, (ax1, ax2) = plt.subplots(1, 2 , figsize=(8,8))
    # Initial Kernel
    ax1.imshow(K_init)
    ax1.set_title('Kernel')
    ax1.axis('off')
    # Reconstruct Kernel
    ax2.imshow(K_alpha)
    ax2.set_title('Reconstruct')
    ax2.axis('off')
    #
    fig2, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(Nxnew, K_i, 'b+', label = "initial")
    ax3.plot(Nxnew, K_a, 'r+', label = "reconstruct")
    ax3.axis(ymin=0)
    ax3.legend()
    # Show plot
    plt.show()

#
def Error_Display(x_init, K, K_alpha):
    """
    Compare blurred image with initial kernel and with reconstruct kernel.
    Compute the error in l2 of the reconstruction.

    Parameters
    ----------
        x_init    (numpy array) : initial inmage
        K         (numpy array) : initial kernel (2Mx2M size)
        K_alpha   (numpy array) : reconstruct kernel
    Returns
    ----------
        -
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15,10))
    # initial image
    ax0.imshow(x_init,cmap='gray')
    ax0.set_title('Initial')
    ax0.axis('off')
    # initial image (blurred)
    x_blurred = Blurr(x_init,K)
    ax1.imshow(x_blurred,cmap='gray')
    ax1.set_title('Blurred')
    ax1.axis('off')
    # reconstruct image (blurred)
    x_blurred_alpha = Blurr(x_init,K_alpha)
    ax2.imshow(x_blurred_alpha,cmap='gray')
    ax2.set_title('Reconstruct Blurred')
    ax2.axis('off')
    # Show plot
    plt.show()
    # Error computation and dispay
    norm     = np.linalg.norm(x_init)
    error_l2 = np.linalg.norm(x_blurred_alpha-x_blurred)/norm
    print("Erreur totale :")
    print(error_l2)