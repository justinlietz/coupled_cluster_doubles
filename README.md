Coupled Cluster code from grad school.
Uses CMake to generate Makefile.
Depends on a c++11 compatible compiler, blas, mpi, and gsl.

General use looks like:
module load gcc
module load openblas
module load openmpi
module load gsl
cmake .
make

This will generate the executable ccd.exe in the src directory.
A readme on running this executable is in src.
