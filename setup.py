from setuptools import setup

setup(name='keris',
      version='0.1',
      description='Keras-like deep learning framework built with numpy',
      url='https://github.com/arrizalamin/keris',
      download_url='https://github.com/arrizalamin/keris/archive/0.1.tar.gz',
      author='Arrizal Amin',
      author_email='arrizalamin@gmail.com',
      license='MIT',
      packages=['keris'],
      keywords=['deep learning', 'keris'],
      install_requires=['numpy>=1.13.3',
                        'tqdm>=4.19.4'])
