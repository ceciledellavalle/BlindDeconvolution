"""
Plotting function to show image, kernel and energies 
---------
    Display_im     : show and compare true image and reconstruct image
    Display_ker    : show and compare true kernel and reconstruct kernel
    Display_epd    : show primal and dual energy through alternating algorithm

@author: Cecile Della Valle
@date: 23/02/2021
"""
# General import
import matplotlib.pyplot as plt
import numpy as np
import math 

def Display_im(x_out,x_true,mysize=(6,4)):
    """
    Compare true image and de-noised de-blurred image with regularization TV.
    Parameters
    ----------
        x_out     (numpy array) : reference, initial or true kernel ( Nx,Ny size)
        x_true    (numpy array) : reconstruct kernel (Nx,Ny size)
    Returns
    ----------
        --
    """
    # defimne graph   
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=mysize)
    # initial image
    ax0.imshow(x_true,cmap='gray')
    ax0.set_title('True')
    ax0.axis('off')
    # Reconstruct image
    ax1.imshow(x_out,cmap='gray')
    ax1.set_title('Reconstruct image')
    ax1.axis('off')
    # Show plot
    plt.show()
    # Error computation and dispay
    norm     = np.linalg.norm(x_true)
    error_l2 = np.linalg.norm(x_out-x_true)/norm
    print("Erreur |x_pred - x_true|_2 :",error_l2)
    
def Display_ker(K_alpha,K_init,mysize=(5,3)):
    """
    Compare initial Kernel and reconstruct Kernel with regularization parameter alpha.
    Parameters
    ----------
        K_alpha   (numpy array) : reconstruct kernel (2Mx2M size)
        K_init    (numpy array) : reference, initial or true kernel (2Mx2M size)
    Returns
    ----------
        -
    """
    # define graph
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3 , figsize=mysize)
    # Initial Kernel
    ax1.imshow(K_init)
    ax1.set_title('Kernel')
    ax1.axis('off')
    # Reconstruct Kernel
    ax2.imshow(K_alpha)
    ax2.set_title('Reconstruct')
    ax2.axis('off')
    # Reconstruct Kernel
    ax3.imshow(K_alpha-K_init)
    ax3.set_title('Comparison')
    ax3.axis('off')
    # Show plot
    plt.show()
    # Error computation and dispay
    norm     = np.linalg.norm(K_init)
    error_l2 = np.linalg.norm(K_alpha-K_init)/norm
    print("Erreur |K_pred - K_true|_2 :",error_l2)
    
def Display_epd(Ep,Ed):
    """
    Compare initial Kernel and reconstruct Kernel with regularization parameter alpha.
    Parameters
    ----------
        Ep    (numpy array) : primal energy 
        Ed    (numpy array) : dual energy
    Returns
    ----------
        -
    """
    # define graph
    fig, (ax1, ax2) = plt.subplots(1, 2 , figsize=(7,7))
    # Initial Kernel
    ax1.plot(Ep,'+')
    ax1.set_title('Primal energy')
    # Reconstruct Kernel
    ax2.plot(Ed,'+')
    ax2.set_title('Dual energy')
    # Show plot
    plt.show()
