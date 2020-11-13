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

    Parameters
    ----------
        x_init (numpy array): image
        K (numpy array): 2D Kernel
    Returns
    -------
        x_blurred (numpy array): output image
    """
    x_blurred = fft_convolve2d(K,x_init)
    return x_blurred
    
def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = fft2(x)
    fr2 = fft2(y)
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, -int(m//2+1),axis=0)
    cc = np.roll(cc, -int(n//2+1),axis=1)
    return cc