"""
Simplex Function
---------
    Simplex     : algorithm to compute the projection onto the canonical simplex
@author: Cecile Della Valle
@date: 09/11/2020
"""

# IMPORTATION
import numpy as np
import sys

# FUNCTION DEFINITION
def Simplex(y):
    """
    Compute the projection onto the canonical simplex. 
    Utilizing the Moreau's identity, the problem is essentially a univariate minimization 
    and the objective function is strictly convex and continuously differentiable. 
    Moreover, it is shown that there are at most n candidates which can be computed explicitly, 
    and the minimizer is the only one that falls into the correct interval. 
    Parameters
    ----------
        y (numpy array): kernel size nxp
    Returns
    -------
        (numpy array): projection of the kernel onto the simplex
    """
    # Step 1 : Sort y in the ascending order
    ## (rearrangment on a 1-D vector)
    y_temp = np.sort(y, axis=None) 
    nsize = len(y_temp)
    # Step 2 : Compute t_i = (\sum_{i}^{n-1} y_j -1)/(n-i)
    ti = (np.cumsum(y_temp[::-1])[::-1] -1)/(nsize-np.arange(0,nsize))
    # Step 3 : Test
    ## If i = 0 then t_hat = (\sum_{0}^{n-1} y_j -1)/n
    ## If t_i ≥ y (i) then set t_hat = t_i
    index = int(np.sum(np.ones(nsize)[ti - y_temp>=0]))
    t_hat = ti[index]  
    # Step 4 : Return x = (y-t_hat)_+ 
    # as the projection of y onto the simplex of size (nxp)
    proj = y - t_hat
    proj[y-t_hat<0]=0
    return proj