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
    
def Display_ker(K_1,K_2,mysize=(5,3),label1='K1',label2='K2'):
    """
    Compare initial Kernel and reconstruct Kernel with regularization parameter alpha.
    Parameters
    ----------
        K_1   (numpy array) : reconstruct kernel (2Mx2M+1 size)
        K_2    (numpy array) : reference, initial or true kernel (2Mx2M+1 size)
    Returns
    ----------
        -
    """
    # define graph
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4 , figsize=mysize,gridspec_kw={'width_ratios': [1,1,1,3]})
    # Initial Kernel
    ax1.imshow(K_1)
    ax1.set_title(label1)
    ax1.axis('off')
    # Reconstruct Kernel
    ax2.imshow(K_2)
    ax2.set_title(label2)
    ax2.axis('off')
    # Reconstruct Kernel
    ax3.imshow(K_1-K_2)
    ax3.set_title('Comparison')
    ax3.axis('off')
    # Comparison of a slice
    ax4.plot(K_1[21,:],label=label1)
    ax4.plot(K_2[21,:],label=label2)
    ax4.legend()
    # Show plot
    plt.show()
    # Error computation and dispay
    norm     = np.linalg.norm(K_2)
    error_l2 = np.linalg.norm(K_1-K_2)/norm
    print("Erreur |"+label1+" - "+label2+"|/ |"+label2+"| : {:.4f} ".format(error_l2))
    
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
