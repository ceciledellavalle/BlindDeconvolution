"""
Tools functions to load, blurr and show images or data
---------
    DataLoader     : Load initial images and preprocess it to be a [Nx x Ny] rectangle
    Blurr          : blur images with kernel K
    Display        : show initial image, blurred image, kernel
@author: Cecile Della Valle
@date: 09/11/2020
"""

# IMPORTATION
import numpy as np
from numpy.fft import fft2, ifft2
import sys
import matplotlib.pyplot as plt
import math
import random
import os
from PIL import Image
from scipy import signal

# 
def DataLoader(file_name,im_name):
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
        K (numpy array): 2D Kernel
    Returns
    -------
        x_blurred (numpy array): output image
    """
    # Convolve
    x_blurred = fft_convolve2d(x_init,K)
    return x_blurred

def Add_noise(x_init,noise_level=0.05):
    """
    Add Gaussian white noise
    Parameters
    ----------
        x_init (numpy array): image
        noise_level (float): level of noise
    Returns
    -------
        x_noise (numpy array): output image
    """
    # Add noise
    m,n = x_init.shape
    x_noise = x_init.copy()
    x_noise += np.median(x_init)*noise_level*(2*np.random.randn(m,n)-1)
    return x_noise
    
def fft_convolve2d(x,y):
    """ 2D convolution, using FFT
    Parameters
    ----------
        x (numpy array): image
        y (numpy array): 2D Kernel
    Returns
    -------
       cc (numpy array): x*y
    """
    # Padd the kernel y
    m,n = x.shape
    p,q = y.shape
    y_padd = np.pad(y, ((m//2-p//2,m//2-p//2),(n//2-q//2,n//2-q//2)), 'constant')
    # Perform fft
    fr = fft2(x)
    fr2 = fft2(y_padd) 
    # Compute multiplication and inverse fft
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, -int(m//2+1),axis=0)
    cc = np.roll(cc, -int(n//2+1),axis=1)
    return cc