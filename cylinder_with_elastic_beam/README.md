# Code validation on FSI benchmark
S. Turek and J. Hron, Proposal for numerical benchmarking of fluid-structure 
interaction between an elastic object and laminar incompressible flow, 
Lecture Notes in Computational Science and Engineering, 2006.


In the beginning of each code problems and comments are written. 

*Total_ALE.py* and uses a classical ALE formulation, with 
the reference configuration bieng the initial configuration from time *t = 0*.
*Total_ALE.py* uses PETSc adaptive time stepping whereas the *Total_ALE.py*
uses self-written theta-scheme.
The arisen system of nonlinear equation is in both cases solved by Newton method.

