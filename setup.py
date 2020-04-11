"""
setup.py
Created on: 10 jan. 2020
Author: Joeran Bosma
"""

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11']

ext_modules = [
               Extension(
                         'Worker',
                         ['main.cpp'],
                         include_dirs=['include'],
                         language='c++',
                         extra_compile_args = cpp_args,
                         ),
               ]

setup(
      name='Metropolis-Hasting MCMC sampler',
      version='0.1.0',
      author='Joeran Bosma',
      author_email='joeran@bosma.co',
      description='Metropolis-Hasting MCMC sampler',
      ext_modules=ext_modules,
      )
