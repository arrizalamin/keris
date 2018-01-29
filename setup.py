from setuptools import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy

# extensions = [
#     Extension('im2col_cython', ['im2col_cython.pyx'],
#               include_dirs=[numpy.get_include()]
#               ),
# ]

# setup(
#     ext_modules=cythonize(extensions),
# )


setup(name='keris',
      version='1.0',
      description='Keras-like deep learning framework built with numpy',
      url='https://github.com/arrizalamin/keris',
      author='Arrizal Amin',
      author_email='arrizalamin@gmail.com',
      license='MIT',
      packages=['keris'],
      install_requires=['numpy>=1.9.1',
                        'tqdm'],
      zip_safe=False)
