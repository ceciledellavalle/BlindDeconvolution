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
    ax0.imshow(x_out,cmap='gray')
    ax0.set_title('image #1')
    ax0.axis('off')
    # Reconstruct image
    ax1.imshow(x_true,cmap='gray')
    ax1.set_title('image #2')
    ax1.axis('off')
    # Show plot
    plt.show()
    # Error computation and dispay
    norm     = np.linalg.norm(x_true)
    error_l2 = np.linalg.norm(x_out-x_true)/norm
    print("Erreur |im1 - im2|/|im2| :{:.4f}".format(error_l2))
    
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
    ax1.set_title('K1')
    ax1.axis('off')
    # Reconstruct Kernel
    ax2.imshow(K_alpha)
    ax2.set_title('K2')
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
    print("Erreur |K1 - K2|/ |K2| : {:.4f} ".format(error_l2))
    
def Display_epd(Ep,Ed):
    """
    Plot energy during optimization, 
    primal energy of kernel k and image u,
    and dual energy of auxiliary variable v.
    Parameters
    ----------
        Ep    (numpy array) : primal energy of kernel and image
        Ed    (numpy array) : dual energy
    Returns
    ----------
        -
    """
    # define graph
    fig, ( ax1, ax2) = plt.subplots(1, 2 , figsize=(8,4))
    # kernel and image
    ax1.plot(Ep,'r')
    ax1.set_title('Primal energy')
    # auxiliary variable
    ax2.plot(Ed)
    ax2.set_title('Dual energy')
    # Show plot
    plt.show()
    
def Display_energy(Ep1,Ep2):
    """
    Plot and compare primal energy during optimization.
    Parameters
    ----------
        Ep1    (numpy array) : primal energy of algo1
        Ep2    (numpy array) : primal energy of algo2
    Returns
    ----------
        -
    """
    # define graph
    fig, (ax1) = plt.subplots(1, 1 , figsize=(5,5))
    # kernel and image
    ax1.plot(Ep1,'r',label='alternate')
    ax1.plot(Ep1,'k',label='PAD')
    ax1.set_title('Primal energy')
    ax1.legend()
    # Show plot
    plt.show()
