from distutils.command.build_ext import build_ext as _build_ext
from distutils.extension import Extension
from distutils.version import LooseVersion
import os.path as op
import pkg_resources
from setuptools import setup, find_packages
import sys
import versioneer

# Key flag that defines if we will attempt to compile native modules or not.
# If False, Menpo will successfully install, just with some features falling
# back to slower pure-Python variants, or being disabled entirely.
#
# If True, Menpo will attempt to compile it's extensions. If this can't be
# done an error will be raised.
COMPILE = False


SETUP_TOOLS_KWARGS = dict(
    name='menpo',
    version=versioneer.get_version(),
    description='A Python toolkit for handling annotated data',
    author='The Menpo Team',
    author_email='menpo-users@googlegroups.com',
    url='http://www.menpo.org',
    package_data={'menpo': ['data/*']},
    tests_require=['nose', 'mock'],
    install_requires=['numpy>=1.10,<1.11',
                      'scipy>=0.17,<0.18',
                      'matplotlib>=1.4,<1.6',
                      'pillow>=3.0,<4.0',
                      'imageio>=1.5.0,<1.6.0'],
    packages=find_packages()  # TODO will this find compiled modules? If it doesn't is that a problem?
)

if sys.version_info.major == 2:
    SETUP_TOOLS_KWARGS['install_requires'].append('pathlib==1.0')


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
    }
]

VERSIONEER_CMDCLASS = versioneer.get_cmdclass()

if not COMPILE:
    print('Installing without compilation')
    setup(cmdclass=VERSIONEER_CMDCLASS, **SETUP_TOOLS_KWARGS)
    sys.exit()

# --------------------------------------------------------------------------- #

# OK, we're compiling.


def path_with_ext(path_tuple, ext):
    dir_path, file_name = path_tuple[:-1], path_tuple[-1]
    p_ext = dir_path + (file_name + '.' + ext, )
    return op.join(*p_ext)


class MenpoExtension(object):

    def __init__(self, pyx_path=None, language="c++", depends=None,
                 sources=None):
        self.pyx_path = pyx_path
        self.language = language
        self.depends = depends if depends is not None else []
        self.sources = [] if sources is not None else []

    @property
    def name(self):
        path, ext = op.splitext(self.pyx_path)
        return path.replace('/', '.')

    @property
    def cpp_path(self):
        path, ext = op.splitext(self.pyx_path)
        return path + ".cpp"

    @property
    def has_been_cythonized(self):
        return op.exists(self.cpp_path)

    def as_cython_precompiled_extension(self):
        return Extension(self.name, [self.cpp_path] + self.sources,
                         language=self.language)

    def as_cythonizable_extension(self):

        return Extension(self.name, self.sources,
                         language=self.language)

EXTENSIONS = [MenpoExtension(**e) for e in EXTENSION_SPECS]

MISSING_EXTENSIONS = list(filter(lambda e: not op.exists(e.pyx_path), EXTENSIONS))
if len(MISSING_EXTENSIONS) > 0:
    print('Error - the following Cython modules are declared in setup.py '
          'but do not exist:')
    print('\n'.join([p.pyx_path for p in MISSING_EXTENSIONS]))
    sys.exit()


from Cython.Build import cythonize
import numpy as np

MINIMUM_CYTHON_VERSION = '0.23.0'

try:
    import Cython
    ver = Cython.__version__
    CYTHON_INSTALLED = ver >= LooseVersion(MINIMUM_CYTHON_VERSION)
except ImportError:
    CYTHON_INSTALLED = False

try:
    if not CYTHON_INSTALLED:
        raise ImportError('No supported version of Cython installed.')
    from Cython.Distutils import build_ext as _build_ext
    cython = True
except ImportError:
    cython = False


class build_ext_with_numpy_dir(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


cython_exts = cythonize(CYTHON_FILES, quiet=True)


SETUP_TOOLS_KWARGS.update(dict(
    build_ext=build_ext_with_numpy_dir,
    ext_modules=cython_exts,
))

setup(**SETUP_TOOLS_KWARGS)
