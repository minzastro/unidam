#!/bin/bash
gfortran -fPIC -c -o Solve_NonLin.o Solve_NonLin.f90 
f2py2 --opt="-O3"  -c -m model_fitter Solve_NonLin.o model_fitter.f90
