import sys
from setuptools import setup, find_packages
import versioneer
from Cython.Build import cythonize
import numpy as np


# ---- C/C++ EXTENSIONS ---- #
cython_modules = ['menpo/shape/mesh/normals.pyx',
                  'menpo/feature/windowiterator.pyx',
                  'menpo/feature/gradient.pyx',
                  'menpo/external/skimage/_warps_cy.pyx',
                  'menpo/image/patches.pyx']

cython_exts = cythonize(cython_modules, quiet=True)
include_dirs = [np.get_include()]
install_requires = ['numpy>=1.10,<1.11',
                    'scipy>=0.17,<0.18',
                    'matplotlib>=1.4,<1.6',
                    'pillow>=3.0,<4.0',
                    'imageio>=1.5.0,<1.6.0',
                    'Cython>=0.23,<0.24']

if sys.version_info.major == 2:
    install_requires.append('pathlib==1.0')

print(find_packages())

setup(name='menpo',
      description='A Python toolkit for handling annotated data',
      author='The Menpo Team',
      author_email='menpo-users@googlegroups.com',
      url='http://www.menpo.org',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      include_dirs=include_dirs,
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=install_requires,
      package_data={'menpo': ['data/*'],
                    '': ['*.pxd', '*.pyx', '*.cpp', '*.h']},
      tests_require=['nose', 'mock']
      )
