from pathlib import Path
import sys

try:
    from setuptools import setup, Extension
    # Required for compatibility with pip (issue #177)
    from setuptools.command.install import install
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.install import install
from distutils.command.build import build
from distutils.command.build_clib import build_clib
from distutils.command.clean import clean  # TODO: make a good clean command
from setuptools.command.develop import develop
from distutils import log
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import versioneer

PY2 = sys.version_info.major == 2


def get_cmdclass():
    cmdclass = versioneer.get_cmdclass()
    _build = cmdclass.get('build', build)  # noqa

    class Build(_build):
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

    class Install(install):
        def run(self):
            self.run_command('build')
            install.run(self)

    class Develop(develop):
        def run(self):
            self.run_command('build')
            develop.run(self)

    cmdclass.update({'build': Build,
                     'build_ext': build_ext,
                     'build_clib': build_clib,
                     'install': Install,
                     'develop': Develop})

    return cmdclass


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
                      'libraries': ['m']})


build_kwds = {
    'include_dirs': include_dirs,
}

ext_modules = cythonize([
    Extension("pysmoothspl._sbart",
              sources=["pysmoothspl/_sbart.pyx"],
              **build_kwds)
])


# Package requirements
extras_require = {
    'base': ['Cython', 'numpy'],
    'test': ['pytest', 'pandas'],
}
if PY2:
    extras_require['base'].append('pathlib')
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

# Get a cmdclass that builds all the things
cmdclass = get_cmdclass()

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
    install_requires=extras_require['base'],
    extras_require=extras_require,
)
