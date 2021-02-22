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
        K      (numpy array): 2D Kernel
    Returns
    -------
        x_blurred (numpy array): output image
    """
    # Padd the kernel y
    m,n   = x_init.shape
    p,q   = K.shape
    K_pad = np.pad(K, ((m//2-p//2,m//2-p//2),(n//2-q//2,n//2-q//2)), 'constant')
    # Perform fft
    fr  = fft2(x_init)
    fr2 = fft2(K_pad) 
    # Compute multiplication and inverse fft
    x_blurred = np.real(ifft2(fr*fr2))
    x_blurred = np.roll(x_blurred, -int(m//2+1),axis=0)
    x_blurred = np.roll(x_blurred, -int(n//2+1),axis=1)
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
    m,n     = x_init.shape
    x_noise = x_init.copy()
    vn      = np.random.randn(m,n)
    vn      = vn/np.linalg.norm(vn)
    x_noise += np.median(x_init)*noise_level*vn
    return x_noise
    
