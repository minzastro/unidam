#!/usr/bin/env python3
import subprocess
from setuptools import find_packages
from numpy.distutils.core import setup, Extension, build_py

model_fitter = Extension('unidam.iso.model_fitter',
                         sources=['unidam/iso/model_fitter.f90'],
                         depends=['unidam/iso/Solve_NonLin.f90'])


class BuildPyCommand(build_py.build_py):
  """Custom build command."""
  def run(self):
    result = subprocess.run(['./build_fortran.sh'], #shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
    for s in [result.stdout, result.stderr]:
        if len(s) > 0:
            print(s)
    build_py.build_py.run(self)


setup(name='unidam',
      version='1.0',
      description='UniDAM tool',
      url='https://github.com/minzastro/unidam',
      author='Alexey Mints',
      author_email='mints@mps.mpg.de',
      packages=find_packages(),
      package_data={'unidam.iso': ['unidam.conf',
                                   'unidam_pdf.conf',
                                   '*.so'],
                    '': ['DEPENDENCIES']},
      include_package_data=True,
      #ext_modules=[model_fitter],
      install_requires=[
          'numpy', 'scipy', 'astropy',
          'simplejson', 'healpy', 'matplotlib',
          'colorlog', 'argparse'
      ],
      scripts=['unidam/iso/unidam_runner.py',
               'unidam/tools/plot_2d_correlations.py',
               'unidam/tools/plot_parts.py',
               'unidam/tools/plot_sed.py',
               'unidam/tools/prepare_data.py'],
      cmdclass={'build_py': BuildPyCommand},
      zip_safe=False)
