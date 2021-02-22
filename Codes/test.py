import numpy as np


# Test simplex method
from simplex import Simplex
a = np.arange(0,10).reshape(2,5)/12
print(a)
b = Simplex(a)
print(b)
print(np.sum(b))