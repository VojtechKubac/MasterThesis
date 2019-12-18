# Code validation on FSI benchmark
S. Turek and J. Hron, Proposal for numerical benchmarking of fluid-structure 
interaction between an elastic object and laminar incompressible flow, 
Lecture Notes in Computational Science and Engineering, 2006.


### Total ALE 
In Total ALE, reference configuration is the initial configuration.
To this we provide three implementations

*Total_ALE.py* - time discretization with fixed time-step size,
Backward Euler or Cranck-Nicolson.
The Finite Elements are discontinuous space for pressure
and bubble-enriched space for velocity and displacement.

*Total_ALE_PETSc_DG_press.py* - PETSc TS time stepping with various
choices for the temporal discretization (see PETSc TS manual).
The same Finite Element space as for *Total_ALE.py*.

*Total_ALE_PETSc_CG_press.py* - the same as *Total_ALE_PETSc_DG_press.py*
except we use continuous pressure and velocity and displacement spaces
without the enrichment. (cheaper version)

### Updated ALE
Reference configuration in each time-step is the configuration computed
in the last time-step.
We made some simplifications (see the Thesis) in order to split the system
into one nonlinear system for velocity and pressure and one linear
system for velocity. This will speed-up the computations

*Updated_ALE.py* - we use the fixed time-step size (in order to use 
PESTc time-stepping we would need to significantly modify the stepper).
The used Finite Elements are the ones with discontinuous pressure.

### Fully Eulerian
This is an attempt to implement Fully Eulerian FSI in FEniCS. Unsuccessful though
(see the reasoning in the Thesis).

*Fully_Eulerian.py* - the main FSI code.

*integration_cutcell.py* - cut integration routines called from *Fully_Eulerian.py*
We believe the cut integration works good.
