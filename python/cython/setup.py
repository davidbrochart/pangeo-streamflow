from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cdelineate.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize("cflood_delineate.pyx"),
    include_dirs=[numpy.get_include()]
)
