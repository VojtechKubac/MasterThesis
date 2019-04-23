# Code validation on FSI benchmark FSI2 
S. Turek and J. Hron, Proposal for numerical benchmarking of fluid-structure 
interaction between an elastic object and laminar incompressible flow, 
Lecture Notes in Computational Science and Engineering, 2006.



**Neither of the codes is complete yet!**    
In the beginning of each code problems and comments are written. 

So far, the only working code (understand the most complete) is *Total_ALE.py*.
*Total_ALE.py* and *Update_ALE.py* use classical ALE formulation, with the difference that
*Total_ALE.py* the initial configuration from time *t = 0* as the reference configuration. 
Whereas *Update_ALE.py* use the configuration computed in previous time step as 
the reference configiration. This means we have to move mesh in each time step.
There are still some problems in the *Update_ALE* approach.

In *Fully_Eulerian.py* integration over cut elements hasn't been overcomed. Here equations for elastic solid are transfered 
to Eulerian coordinates, this yields to movement of elastic solid over static mesh.
There are many subroutines dealing with integration of cut elements, this will certainly be cleaned some day.

The last code which is in progress is *CutFEM.py*. Here two overlapping meshes are present, 
FEniCS has *multimesh* concept for this. This allows us to
treat each problem in its natural configuration, elasticity in Lagrangian and Navier-Stokes equations in Eulerian.
The problem here is the need of usage of different FEniCS routines for *multimesh* problems.

Finally, *utils.py* is used for loading meshes from directory *meshes* and giving benchmark specific constants.
