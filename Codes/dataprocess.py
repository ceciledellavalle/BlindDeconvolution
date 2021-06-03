"""
Tools functions to load, blurr and show images or data
---------
    DataLoader     : Load initial images and preprocess it to be a [Nx x Ny] rectangle
    Blurr          : blur images with kernel K
    Add_noise      : add gaussian white noise to an image
@author: Cecile Della Valle
@date: 09/11/2020
"""

# IMPORTATION
import numpy as np
from PIL import Image
import imageio
import sys
import matplotlib.pyplot as plt
import math
import random
import os
from scipy import signal
from scipy.interpolate import interp1d
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
# local import
from Codes.simplex import Simplex
from Codes.display import Display_ker
from Codes.myfunc import convolve

#
def DataGen(M=20,gauss=(0.2, 0.0),noise=0.1):
    #================================================================
    # KERNEL
    #================================================================
    # Kernel generator
    gridx, gridy  = np.meshgrid(np.linspace(-1,1,2*M+1), np.linspace(-1,1,2*M+1))
    gd            = np.sqrt(gridx*gridx+gridy*gridy)
    sigma,moy     = gauss
    K             = np.exp(-( (gd-moy)**2 / ( 2.0 * sigma**2 ) ) )
    K             = K/np.sum(K)
    # K_shift
    sigma_shift   = sigma + 0.05
    K_shift       = np.exp(-( (gd-moy)**2 / ( 2.0 * sigma_shift**2 ) ) )
    K_shift       = K_shift/np.sum(K_shift)
    # plot
    Display_ker(K_shift,K,mysize=(10,4),label1='Shift',label2='True')
    #================================================================
    # IMAGE
    #================================================================
    # Shepp-logan image generator
    image        = shepp_logan_phantom()
    image        = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    x_im         = 0.99*image/np.amax(image)
    x_im[x_im<0] = 0
    x_im         = np.pad(x_im, ((9,10),(2,3)), 'constant')
    # Blurred
    x_blurr      = Blurr(x_im,K)
    # Add noise
    x_noisy      = Add_noise(x_blurr,noise_level=noise)
    # plot
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9,3)) # defimne graph  
    # initial image
    ax0.imshow(x_im,cmap='gray')
    ax0.set_title('init')
    ax0.axis('off')
    # blurred
    ax1.imshow(x_blurr,cmap='gray')
    ax1.set_title('blurred')
    ax1.axis('off')
    # blurred&noisy
    ax2.imshow(x_noisy,cmap='gray')
    ax2.set_title('noisy')
    ax2.axis('off')
    # Show plot
    plt.show()
    # Error computation and dispay
    norm     = np.linalg.norm(x_im)
    err_bl2 = np.linalg.norm(x_blurr-x_im)/norm
    err_nl2 = np.linalg.norm(x_noisy-x_im)/norm
    print("Erreur blurred |x_blurr- x_true|_2 :{:.4f}".format(err_bl2))
    print("Erreur |x_noisy - x_true|_2 :{:.4f}".format(err_nl2))
    # return
    return K, K_shift, x_im, x_blurr, x_noisy
    
# Export energy to compare methods
def Export_ep(Ep,label='1',name='nom'):
    """
        Save a function in a chose folder
        for plot purpose.

        Parameters
        ----------
            Ep    (numpy array): path of the folder containing images
            label      (string): '1' correspond to alternate and '2' to pda (algoviolet)
            cas        (string): other name
        Returns
        -------
            --
    """
    # initialisation
    folder ='./Redaction/data' #fichier
    Npoint = np.size(Ep)
    xdata  = np.linspace(0,Npoint-1,Npoint)
    ydata  = Ep.copy()
    # interpolation
    f = interp1d(xdata, ydata)
    xnew = np.linspace(0,Npoint-1,100)
    ynew = f(xnew)
    # name
    if label=='1':
        name='alt'+name
    if label=='2':
        name='pda'+name
    with open(folder+'/'+name+'.txt', 'w') as f:
        for i in range(100):
            web_browsers = ['{0}'.format(xnew[i]),' ','{0} \n'.format(ynew[i])]
            f.writelines(web_browsers)
                
# Export image
def Export_im(g,label='im'):
    """
        Save an image in a chose folder
        for plot purpose.

        Parameters
        ----------
            g    (numpy array): kernel or image
            label     (string): name of the image
        Returns
        -------
            --
    """
    imageio.imwrite(label+'.jpg', x_b)

              
# Image loader
def ImLoader(file_name,im_name):
    """
    Load an image from path, crop it with even dimension Nx, Ny
    If colored images, keep only one chanel

    Parameters
    ----------
        path (string): path of the folder containing images
    Returns
    -------
        x_init (numpy array): image
    """
    # Load image
    im = Image.open(os.path.join(file_name,im_name))
    # Transform in nparray
    data = np.asarray(im)
    # Return only one channel
    return data[:,:,0]

def Blurr(x_init,K):
    """
    Compute the convolution of an image with a Kernel of size (2M)^2
    Add Gaussian white noise
    Parameters
    ----------
        x_init (numpy array): image
        K      (numpy array): 2D Kernel
    Returns
    -------
        x_blurred (numpy array): output image
    """
    # Padd the kernel y
    Nx,Ny = x_init.shape
    M,_   = K.shape
    M     = M//2
    K_pad = np.pad(K, ((Nx//2-M,Nx//2-M),(Ny//2-M,Ny//2-M)), 'constant')
    # Compute convolution
    x_blurred = convolve(x_init,K_pad)
    return x_blurred

def Add_noise(x_init,noise_level=0.01):
    """
    Add Gaussian white noise
    Parameters
    ----------
        x_init (numpy array): image
        noise_level  (float): level of noise
    Returns
    -------
        x_noise (numpy array): output image
    """
    # Add noise
    m,n      = x_init.shape
    x_noise  = x_init.copy()
    vn       = np.random.randn(m,n)
    vn       = vn/np.linalg.norm(vn)
    x_noise += np.linalg.norm(x_init)*noise_level*vn
    return x_noise
    
