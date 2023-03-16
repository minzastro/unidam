#!/usr/bin/env python3
import subprocess
from setuptools import find_packages
from numpy.distutils.core import setup, Extension, build_py

model_fitter = Extension('unidam.core.model_fitter',
                         sources=['unidam/core/model_fitter.f90',
                                  'unidam/core/Solve_NonLin.f90'])

extra_functions = Extension('unidam.utils.extra_functions',
                            sources=['unidam/utils/extra_functions.f90'])


class BuildPyCommand(build_py.build_py):
    """Custom build command."""
    def run(self):
        result = subprocess.run(['./build_fortran.sh'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
        for s in [result.stdout, result.stderr]:
            if len(s) > 0:
                print(s)
        build_py.build_py.run(self)


setup(name='unidam',
      version='2.4',
      description='UniDAM tool',
      url='https://github.com/minzastro/unidam',
      author='Alexey Mints',
      author_email='amints@aip.de',
      packages=find_packages(),
      package_data={'unidam.core': ['unidam.conf',
                                    'unidam_pdf.conf',
                                    '*.so'],
                    '': ['DEPENDENCIES']},
      include_package_data=True,
      install_requires=[
          'numpy', 'scipy', 'astropy',
          'simplejson', 'healpy', 'matplotlib',
          'colorlog', 'argparse'
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      scripts=['unidam/core/unidam_runner.py',
               'unidam/tools/plot_2d_correlations.py',
               'unidam/tools/plot_parts.py',
               'unidam/tools/plot_sed.py',
               'unidam/tools/prepare_data.py'],
      cmdclass={'build_py': BuildPyCommand},
      ext_modules=[model_fitter, extra_functions],
      zip_safe=False)
