from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("SENSE_module.pyx")
)

ext_modules=[ Extension("SENSE_module",
              ["SENSE_module.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "SENSE_module",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
