from distutils.core import Command
from distutils.extension import Extension
from distutils.version import LooseVersion
from distutils.command.sdist import sdist as default_sdist
import os.path as op
import pkg_resources


def is_cython_installed(min_version=None):
    try:
        import Cython
        if min_version is not None:
            return Cython.__version__ >= LooseVersion(min_version)
        else:
            return True
    except ImportError:
        return False


def default_build_ext_factory():
    from distutils.command.build_ext import build_ext as default_build_ext
    return default_build_ext


def cython_build_ext_factory():
    from Cython.Distutils import build_ext as cython_build_ext
    return cython_build_ext


def build_ext_with_numpy_include_dir_factory(supercls):

    class BuildExtWithNumpyIncludeDir(supercls):

        def build_extensions(self):
            numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

            for ext in self.extensions:
                if hasattr(ext, 'include_dirs') and numpy_incl not in ext.include_dirs:
                    ext.include_dirs.append(numpy_incl)
            supercls.build_extensions(self)

    return BuildExtWithNumpyIncludeDir


def build_ext_without_compile_factory(supercls):

    class BuildExtWithoutCompile(supercls):
        """Custom distutils command subclassed from Cython.Distutils.build_ext
        to compile .pyx->.c/.cpp, and stop there. All this does is override the
        actual compile method build_extension() with a no-op."""
        def build_extension(self, ext):
            pass

    return BuildExtWithoutCompile


def build_ext_allowing_failure_factory(supercls):

    class BuildExtAllowingFailure(supercls):
        """Subclass that allows for extensions to be optional and to fail to
        install.
        """
        def build_extensions(self):
            try:
                supercls.build_extensions(self)
            except Exception as e:
                print('Failed to build extensions due to error:')
                print(e)
                print('Skipping as this is an optional extension.')

    return BuildExtAllowingFailure


class CythonExtension(object):

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
    def native_suffix(self):
        return ".cpp" if self.language == "c++" else ".c"

    @property
    def cythonized_path(self):
        path, ext = op.splitext(self.pyx_path)
        return path + self.native_suffix

    @property
    def has_been_cythonized(self):
        return op.exists(self.cythonized_path)

    def as_already_cythonized_extension(self):
        return Extension(self.name, [self.cythonized_path] + self.sources,
                         language=self.language)

    def as_cythonizable_extension(self):
        return Extension(self.name, self.sources,
                         language=self.language)


def check_extensions_are_valid(extensions):
    missing_exts = list(filter(lambda e: not op.exists(e.pyx_path), extensions))
    if len(missing_exts) > 0:
        message = ('The following Cython modules are declared in setup.py '
                   'but do not exist:\n{}'.format('\n'.join([p.pyx_path for p in missing_exts])))
        raise ValueError(message)


def extensions_needing_cython(extensions):
    return list(filter(lambda e: not e.has_been_cythonized, extensions))


def ext_setup_kwargs(extension_specs, cmdclass=None, min_cython_version=None, failure_permitted=False):

    cmdclass = cmdclass if cmdclass is not None else {}
    sdist_base_class = cmdclass.get('sdist', default_sdist)

    extensions = [CythonExtension(**e) for e in extension_specs]
    check_extensions_are_valid(extensions)
    cython_installed = is_cython_installed(min_version=min_cython_version)

    ext_needing_cython = extensions_needing_cython(extensions)
    cython_required = len(ext_needing_cython) != 0

    if cython_required:
        if not cython_installed:
            message = ('Cython is required to process the following extensions '
                       'but it is not installed:\n{}'.format(
                          '\n'.join([p.pyx_path for p in ext_needing_cython])))
            raise ValueError(message)
        else:
            message = ('Cython will be used to process the following extensions:'
                       '\n{}'.format('\n'.join([p.pyx_path for p in ext_needing_cython])))
            print(message)
            from Cython.Build import cythonize
            ext_modules = cythonize([e.as_cythonizable_extension()
                                     for e in ext_needing_cython], quiet=False)
            print('finished cythonizing')
    else:
        # Everything has been Cythonized for us - just need to build the modules
        # directly.
        ext_modules = [e.as_already_cythonized_extension() for e in extensions]

    # We always start from the default build ext and add the numpy
    # include dir
    build_ext = build_ext_with_numpy_include_dir_factory(
        default_build_ext_factory())

    if failure_permitted:
        build_ext = build_ext_allowing_failure_factory(build_ext)

    cmdclass["build_ext"] = build_ext

    return {
        "cmdclass": cmdclass,
        "ext_modules": ext_modules
    }
