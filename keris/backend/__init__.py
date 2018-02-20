import sys
from .ndarray import *

_BACKEND = 'numpy'

if _BACKEND == 'numpy':
    sys.stderr.write('Using numpy backend\n')
    from numpy import *


def backend():
    return _BACKEND
