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
            print(ext)
            print('we are building the cython ext even though its sdist!')
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


def s_dist_invoking_cython_factory(supercls):

    class SDistInvokingCython(supercls):
        """Custom sdist that ensures Cython has compiled all pyx files to .c/.cpp"""

        def run(self):
            print('about to run cython command')
            self.run_command('cython')
            print('cython command run.')
            supercls.run(self)

    return SDistInvokingCython


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
            # whether we are doing a build_ext or sdist/bdist, we have to
            # ensure that Cython (which is available) is invoked to generate
            # the required .c/.cpp extensions.
            message = ('Cython will be used to process the following extensions:'
                       '\n{}'.format('\n'.join([p.pyx_path for p in ext_needing_cython])))
            print(message)

            # prepare a build_ext that will just compile the pyx files to
            # .c/.cpp files and stop.
            cythonize_and_stop_build_ext = build_ext_without_compile_factory(
                build_ext_with_numpy_include_dir_factory(
                    cython_build_ext_factory()
                    )
                )

            # add this to the cmdclass dict so it is available at sdist time
            cmdclass['cython'] = cythonize_and_stop_build_ext
            # now add an sdist command which will invoke cython when we try
            # to build an sdist.
            cmdclass['sdist'] = s_dist_invoking_cython_factory(sdist_base_class)

            # make sure the ext modules are ready for processing by the
            # cythonize compiler.
            ext_modules = [e.as_cythonizable_extension()
                           for e in ext_needing_cython]
            print('finished cythonizing')
    else:
        # Everything has been Cythonized for us - no need for Cython we
        # just need to build the modules directly.
        # note that this also means we are happy to do a sdist command in a
        # 'standard' fashion.
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
