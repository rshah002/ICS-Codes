from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("XSENSE_module.pyx")
)

ext_modules=[ Extension("XSENSE_module",
              ["XSENSE_module.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "XSENSE_module",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
