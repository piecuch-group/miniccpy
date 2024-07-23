from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["cc3.pyx", "rcc3.pyx"],
                          annotate=True),
    include_dirs=[numpy.get_include()],
)
