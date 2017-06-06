from pathlib import Path
import sys

from distutils.command.build import build
from distutils.command.build_clib import build_clib
from distutils.command.clean import clean
from distutils import log
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import setup, Extension

import versioneer

cmdclass = versioneer.get_cmdclass()
build = cmdclass.get('build', build)


class Build(build):
    """ A build class that also does `build_clib` & `build_ext`
    """
    def finalize_options(self):
        build.finalize_options(self)
        # The build isn't platform-independent
        if self.build_lib == self.build_purelib:
            self.build_lib = self.build_platlib

    def get_sub_commands(self):
        # Force "build_clib"/"build_ext" invocation.
        commands = build.get_sub_commands(self)
        if 'build_ext' not in commands:
            commands.insert(0, 'build_ext')
        if 'build_clib' not in commands:
            commands.insert(0, 'build_clib')
        return commands


# Include src/
srcdir = Path('src/')
include_dirs = [str(srcdir)]

# Need NumPy headers
try:
    import numpy as np
    include_dirs.append(np.get_include())
except ImportError:
    log.critical("Numpy and its headers are required to run install.")
    sys.exit(1)


# Define sbart.c as a library so we can compile it before our wrapper
sources = list(map(str, srcdir.glob('*.c')))
libsbart = ('sbart', {'sources': sources,
                      'include_dirs': [str(srcdir)],
                      'libraries': ['m', 'gfortran']})


build_kwds = {
    'include_dirs': include_dirs
}

ext_modules = cythonize([
    Extension("pysmoothspl._sbart",
              sources=["pysmoothspl/_sbart.pyx"],
              **build_kwds)
])

cmdclass.update({'build': Build,
                 'build_ext': build_ext,
                 'build_clib': build_clib})


setup(
    name="pysmoothspl",
    version=versioneer.get_version(),
    author="Chris Holden",
    author_email="ceholden@gmail.com",
    url="http://github.com/ceholden/pysmoothspl",
    license="GPLv3",
    packages=["pysmoothspl"],
    libraries=[libsbart],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
