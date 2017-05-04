#!/usr/bin/env python2
from setuptools import find_packages
from numpy.distutils.core import setup, Extension

model_fitter = Extension('unidam.iso.model_fitter',
                         sources=['unidam/iso/model_fitter.f90'])

setup(name='unidam',
      version='1.0',
      description='UniDAM tool',
      url='https://github.com/minzastro/unidam',
      author='Alexey Mints',
      author_email='mints@mps.mpg.de',
      packages=find_packages(),
      package_data={'unidam.iso': ['unidam.conf',
                                   'unidam_pdf.conf'],
                    '': ['DEPENDENCIES']},
      include_package_data=True,
      ext_modules=[model_fitter],
      install_requires=[
          'numpy', 'scipy', 'astropy',
          'simplejson', 'healpy', 'matplotlib',
          'configparser', 'colorlog', 'argparse'
      ],
      scripts=['unidam/iso/unidam_runner.py'],
      zip_safe=False)