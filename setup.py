import os
os.environ['PATH'] = '/Users/jab08/miniconda/envs/optcython/bin:/Users/jab08/.nvm/versions/node/v5.11.0/bin:/Applications/Postgres.app/Contents/Versions/latest/bin:/Users/jab08/.bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/usr/local/go/bin:/Library/TeX/texbin'

from setuptools import setup, find_packages
import sys
import versioneer
from extmanager import ext_setup_kwargs

# Key flag that defines if we will attempt to compile native modules or not.
# If False, Menpo will successfully install, just with some features falling
# back to slower pure-Python variants, or being disabled entirely.
#
# If True, Menpo will attempt to compile it's extensions. If this can't be
# done an error will be raised.
ATTEMPT_TO_COMPILE = True


EXTENSION_SPECS = [
    {
        "pyx_path": "menpo/external/skimage/_warps_cy.pyx"
    },
    {
        "pyx_path": "menpo/feature/gradient.pyx"
    },
    {
        "pyx_path": "menpo/feature/windowiterator.pyx",
        "depends": [
            "menpo/feature/cpp/HOG.h",
            "menpo/feature/cpp/ImageWindowIterator.h",
            "menpo/feature/cpp/LBP.h",
            "menpo/feature/cpp/WindowFeature.h"
        ],
        "sources": [
            "menpo/feature/cpp/ImageWindowIterator.cpp",
            "menpo/feature/cpp/WindowFeature.cpp",
            "menpo/feature/cpp/HOG.cpp",
            "menpo/feature/cpp/LBP.cpp"
        ]
    },
    {
        "pyx_path": "menpo/image/patches.pyx",
    },
    {
        "pyx_path": "menpo/shape/mesh/normals.pyx",
        "language": "c"
    }
]


SETUP_TOOLS_KWARGS = dict(
    name='menpo',
    version=versioneer.get_version(),
    description='A Python toolkit for handling annotated data',
    author='The Menpo Team',
    author_email='menpo-users@googlegroups.com',
    url='http://www.menpo.org',
    package_data={'menpo': ['data/*']},
    tests_require=['nose', 'mock'],
    install_requires=['numpy>=1.9.1',
                      'scipy>=0.14',
                      'matplotlib>=1.4,<1.6',
                      'pillow>=3.0,<4.0',
                      'imageio>=1.5.0,<1.6.0'],
    setup_requires=['numpy>=1.9.1'],
    packages=find_packages(),  # TODO will this find compiled modules? If it doesn't is that a problem?
)

if sys.version_info.major == 2:
    SETUP_TOOLS_KWARGS['install_requires'].append('pathlib==1.0')

VERSIONEER_CMDCLASS = versioneer.get_cmdclass()

if not ATTEMPT_TO_COMPILE:
    print('Installing without compilation')
    SETUP_TOOLS_KWARGS["cmdclass"] = VERSIONEER_CMDCLASS
else:
    print('Attempting to compile extension modules')
    SETUP_TOOLS_KWARGS.update(ext_setup_kwargs(EXTENSION_SPECS,
                                               cmdclass=VERSIONEER_CMDCLASS,
                                               min_cython_version='0.23.0'))

setup(**SETUP_TOOLS_KWARGS)
