import sys
import os
from .ndarray import *

_BACKEND = 'numpy'

if 'KERIS_BACKEND' in os.environ:
    _BACKEND = os.environ['KERIS_BACKEND']

if _BACKEND == 'numpy':
    sys.stderr.write('Using numpy backend\n')
    from numpy import *
elif _BACKEND == 'cupy':
    sys.stderr.write('Using cupy backend\n')
    from cupy import *


def backend():
    return _BACKEND
