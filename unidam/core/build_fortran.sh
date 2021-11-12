#!/bin/bash
gfortran -fPIC -c -o Solve_NonLin.o Solve_NonLin.f90 
f2py --opt="-O3"  -c -m model_fitter Solve_NonLin.o model_fitter.f90
cd ../utils
f2py --opt="-O3"  -c -m extra_functions extra_functions.f90
cd -