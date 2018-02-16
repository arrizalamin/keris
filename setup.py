from setuptools import setup
from distutils.extension import Extension
import numpy

extensions = [
    Extension('keris.utils.im2col',
              ['keris/utils/im2col.c'],
              include_dirs=[numpy.get_include()]),
]

setup(name='keris',
      version='0.3',
      description='Simple keras-like deep learning framework built with numpy',
      url='https://github.com/arrizalamin/keris',
      download_url='https://github.com/arrizalamin/keris/archive/0.3.tar.gz',
      author='Arrizal Amin',
      author_email='arrizalamin@gmail.com',
      license='MIT',
      packages=['keris'],
      ext_modules=extensions,
      keywords=['deep learning', 'keris'],
      install_requires=['numpy>=1.13.3',
                        'tqdm>=4.19.4'])
