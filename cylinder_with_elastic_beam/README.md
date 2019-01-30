# Code validation on FSI benchmark FSI2 
S. Turek and J. Hron, Proposal for numerical benchmarking of fluid-structure 
interaction between an elastic object and laminar incompressible flow, 
Lecture Notes in Computational Science and Engineering, 2006.



**Neither of the codes is complete yet!**    
In the beginning of each code problems and comments are written. 

So far the best working codes (understand the most complete) are *Update_ALE.py* and *Full_ALE.py*, 
these ones uses the classical ALE formulations.
*Full_ALE.py* works in initial configuration from time *t = 0*, whereas *Update_ALE.py* moves mesh in each time iteration
so the Lagrangian configuration for current time step is the one computed in precious time step.

In *Fully_Eulerian.py* integration over cut elements hasn't been overcomed. Here equations for elastic solid are transfered 
to Eulerian coordinates, this yields to movement of elastic solid over static mesh.
There are many subroutines dealing with integration of cut elements, this will certainly be cleaned some day.

The last code which is in progress is *CutFEM.py*. Here two overlapping meshes are present, 
FEniCS has *multimesh* concept for this. This allows us to
treat each problem in its natural configuration, elasticity in Lagrangian and Navier-Stokes equations in Eulerian.
The problem here is the need of usage of different FEniCS routines for *multimesh* problems.

Finally, *marker.py* is used for generating meshes and marking subdomains and boundaries.
