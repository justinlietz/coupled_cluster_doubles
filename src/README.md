// Copyright (c) 2015-2022, Justin Gage Lietz
// All rights reserved.

To run on home PC, need to use cblas rather than mkl.

Minimal example run:

./main.exe 1 0.08 1 3 14 1.e-8

should get E_CCD/A = 10.155078

The input parameters mean:
1 = calculate infinite matter with minnesota potential
0.08 = density of infinite matter
1 = neutrons only (2 for neutrons and protons)
3 = number of single particle momentum shells
14 = number of neutrons in calculation (must be magic numbers)
1.e-8 = difference tolerance of energy for iterative solver
