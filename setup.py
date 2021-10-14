from __future__ import print_function
from future.utils import iteritems
from builtins import next
from builtins import filter
from Cython.Build import cythonize

import os
from os.path import join as pjoin
from glob import glob
from ast import parse
from setuptools import setup
from distutils.extension import Extension
from distutils.command.clean import clean
from Cython.Distutils import build_ext
import subprocess
import re
import numpy
from distutils.spawn import spawn, find_executable
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import sys

#change for windows
nvcc_bin = 'nvcc.exe'
PATH = os.environ.get('PATH')

# The approach used in this file is copied from the cython/CUDA setup.py
# example at https://github.com/rmcgibbo/npcuda-example
def find_in_path(name, path):
    "Find a file in a search path"

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin',nvcc_bin)
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home,'lib','x64')}
    for k, v in iteritems(cudaconfig):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

def get_cuda_version(cucnf):
    """Get the installed CUDA version from nvcc"""

    try:
        nvcc = cucnf['nvcc']
        verstr = subprocess.check_output([nvcc, '-V']).decode("utf-8")
        m = re.search('release (\d+\.\d+)', verstr)
        ver = m.group(1)
    except:
        ver = None

    return ver

# Run the customize_compiler
class custom_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        if hasattr(self.compiler, '_c_extensions'):
            self.compiler._c_extensions.append('.cu')  # needed for Windows
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        elif self.compiler.compiler_type == 'msvc':
            # There are several things we need to do to change the commands
            # issued by MSVCCompiler into one that works with nvcc. In the end,
            # it might have been easier to write our own CCompiler class for
            # nvcc, as we're only interested in creating a shared library to
            # load with ctypes, not in creating an importable Python extension.
            # - First, we replace the cl.exe or link.exe call with an nvcc
            #   call. In case we're running Anaconda, we search cl.exe in the
            #   original search path we captured further above -- Anaconda
            #   inserts a MSVC version into PATH that is too old for nvcc.
            cmd[:1] = ['nvcc', '--compiler-bindir',
                       os.path.dirname(find_executable("cl.exe", PATH))
                       or cmd[0]]
            # - Secondly, we fix a bunch of command line arguments.
            for idx, c in enumerate(cmd):
                # create .dll instead of .pyd files
                if '.pyd' in c: cmd[idx] = c = c.replace('.pyd', '.dll')
                # replace /c by -c
                if c == '/c': cmd[idx] = '-c'
                # replace /DLL by --shared
                elif c == '/DLL': cmd[idx] = '--shared'
                # remove --compiler-options=-fPIC
                elif '-fPIC' in c: del cmd[idx]
                # replace /Tc... by ...
                elif c.startswith('/Tc'): cmd[idx] = c[3:]
                # replace /Fo... by -o ...
                elif c.startswith('/Fo'): cmd[idx:idx+1] = ['-o', c[3:]]
                # replace /LIBPATH:... by -L...
                elif c.startswith('/LIBPATH:'): cmd[idx] = '-L' + c[9:]
                # replace /OUT:... by -o ...
                elif c.startswith('/OUT:'): cmd[idx:idx+1] = ['-o', c[5:]]
                # remove /EXPORT:initlibcudamat or /EXPORT:initlibcudalearn
                elif c.startswith('/EXPORT:'): del cmd[idx]
                # replace cublas.lib by -lcublas
                elif c == 'cublas.lib': cmd[idx] = '-lcublas'
            # - Finally, we pass on all arguments starting with a '/' to the
            #   compiler or linker, and have nvcc handle all other arguments
            if '--shared' in cmd:
                pass_on = '--linker-options='
                # we only need MSVCRT for a .dll, remove CMT if it sneaks in:
                cmd.append('/NODEFAULTLIB:libcmt.lib')
            else:
                pass_on = '--compiler-options='
            cmd = ([c for c in cmd if c[0] != '/'] +
                   [pass_on + ','.join(c for c in cmd if c[0] == '/')])
            # For the future: Apart from the wrongly set PATH by Anaconda, it
            # would suffice to run the following for compilation on Windows:
            # nvcc -c -O -o <file>.obj <file>.cu
            # And the following for linking:
            # nvcc --shared -o <file>.dll <file1>.obj <file2>.obj -lcublas
            # This could be done by a NVCCCompiler class for all platforms.
        spawn(cmd, search_path, verbose, dry_run)

class custom_clean(clean):
    def run(self):
        super(custom_clean, self).run()
        for f in glob(os.path.join('sporco_cuda', '*.pyx')):
            os.unlink(os.path.splitext(f)[0] + '.c')
        for f in glob(os.path.join('sporco_cuda', '*.pyd')):
            os.unlink(f)

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


CUDA = locate_cuda()
cuver = get_cuda_version(CUDA)

# Default CUDA version if version number cannot be extracted from nvcc
if cuver is None:
    cuver = '10.0'

cc = [
    '-gencode', 'arch=compute_30,code=sm_30',
    '-gencode', 'arch=compute_32,code=sm_32',
    '-gencode', 'arch=compute_35,code=sm_35',
    '-gencode', 'arch=compute_37,code=sm_37',
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_53,code=sm_53',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_62,code=sm_62',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_72,code=sm_72',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86'
]
if cuver == '8.0':
    archflg = cc[0:20]
elif cuver == '9.0' or cuver == '9.1' or cuver == '9.2':
    archflg = cc[0:24]
elif cuver == '10.0' or cuver == '10.1' or cuver == '10.2':
    archflg = cc[0:26]
elif cuver == '11.0':
    archflg = cc[4:28]
elif cuver == '11.1' or cuver == '11.2' or cuver == '11.3' or cuver == '11.4':
    archflg = cc[4:30]
else:
    archflg = cc[8:30]


# M75 or SM_75, compute_75 — GTX Turing — GTX 1660 Ti
nvcc_flags = ['-O', '--ptxas-options=-v', '-arch=sm_75', '-c', '--compiler-options=-fPIC']
ext_util = Extension('sporco_cuda.util',
        sources= ['sporco_cuda/src/utils.cu',
                  'sporco_cuda/util.pyx'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cuda', 'cudart'],
        language = 'c',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args = {
            'gcc': [],
            'nvcc': nvcc_flags
            },
        include_dirs = [numpy_include, CUDA['include'], 'sporco_cuda/src'])

ext_cbpdn = Extension('sporco_cuda.cbpdn',
        sources= ['sporco_cuda/src/utils.cu',
                  'sporco_cuda/src/cbpdn_kernels.cu',
                  'sporco_cuda/src/cbpdn.cu',
                  'sporco_cuda/src/cbpdn_grd.cu',
                  'sporco_cuda/cbpdn.pyx'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cuda', 'cudart', 'cufft', 'cublas', 'm'],
        language = 'c',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args = {
            'gcc': [],
            'nvcc': nvcc_flags
            },
        include_dirs = [numpy_include, CUDA['include'], 'sporco_cuda/src'])

name = 'sporco-cuda'
pname = 'sporco_cuda'

# Get version number from sporco_cuda/__init__.py
# See http://stackoverflow.com/questions/2058802
with open(os.path.join(pname, '__init__.py')) as f:
    version = parse(next(filter(
        lambda line: line.startswith('__version__'),
        f))).body[0].value.s


longdesc = \
"""
SPORCO-CUDA is an extension package to Sparse Optimisation Research
Code (SPORCO), providing GPU accelerated versions for some
convolutional sparse coding problems.
"""

setup(
    author           = 'Gustavo Silva, Brendt Wohlberg',
    author_email     = 'gustavo.silva@pucp.edu.pe, brendt@ieee.org',
    name             = name,
    description      = 'SPORCO-CUDA: A CUDA extension package for SPORCO',
    long_description = longdesc,
    keywords         = ['Convolutional Sparse Representations',
                        'Convolutional Sparse Coding', 'CUDA'],
    url              = 'https://github.com/bwohlberg/sporco-cuda',
    version          = version,
    platforms        = 'Linux',
    license          = 'BSD',
    setup_requires   = ['cython', 'future', 'numpy'],
    #tests_require    = ['pytest', 'pytest-runner', 'sporco'],
    tests_require    = ['pytest', 'pytest-runner'],
    install_requires = ['future', 'numpy'],
    classifiers = [
    'License :: OSI Approved :: BSD License',
    'Development Status :: 4 - Beta',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules'
    ],

    # extension module specification
    ext_modules = cythonize([ext_util, ext_cbpdn],compiler_directives={'language_level' : "3"}),
    # inject our custom trigger
    cmdclass = {'build_ext': custom_build_ext,
                'clean': custom_clean},
    # since the package has c code, the egg cannot be zipped
    zip_safe = False)


