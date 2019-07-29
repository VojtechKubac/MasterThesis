"""
This is part of Master Thesis by Vojtech Kubac.

This code solves FSI2 and FSI3 Bencmarks from
    "S. Turek and J. Hron, “Proposal for numerical benchmarking of fluid–
     structure interaction between an elastic object and laminar incompress-
     ible flow,” in Fluid-Structure Interaction - Modelling, Simulation, Opti-
     mization, ser. Lecture Notes in Computational Science and Engineering,"

The equations are written in Total-ALE formulation, where for the mesh movement pseudoelasticity
extension of the solid displacement was used.

Chosen Finite Elements are linear discontinuous space for pressure and quadratic continuous space
enriched with quadratic bubble is used for displacement and velocity.

Time discretization uses PETSc time stepping methods, preferably BDF, BE and CN.
"""

from dolfin import *
from dolfin import __version__
import numpy as np
import csv
import sys
import os.path
from mpi4py.MPI import COMM_WORLD
from optparse import OptionParser

import os
import gc

sys.path.append(os.getcwd())
sys.path.append('../petscsolvers/')
sys.path.append('.')

import matplotlib.pyplot as plt
import mpi4py
import mpi4py.MPI
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from fenics_utils import Report
import petsc_ts_solver_for_Total_ALE as TS


# Define MPI World
if __version__[:4] == '2018' or  __version__[:4] == '2019':
    comm = MPI.comm_world
else:
    comm = mpi_comm_world()
my_rank = comm.Get_rank()

# Use UFLACS to speed-up assembly and limit quadrature degree
parameters["std_out_all_processes"] = False
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4
ffc_opts = {"quadrature_degree": 4, "optimize": True}

parameters['ghost_mode'] = 'shared_facet'

PETScOptions.set('mat_mumps_icntl_24', 1)		# detects null pivots
PETScOptions.set('mat_mumps_cntl_1', 0.01)		# set treshold for partial treshold pivoting, 0.01 is default value

opts = PETSc.Options()
opts['options_left'] = True
opts.setFromOptions()

class Flow(object):
    """
    Class where the equations for the FSI are defined. It possesses methods 'solve' and 'save'
    that solves equations in each time step and  then saves the obtained results.
    """
    def __init__(self, mesh, bndry, interface, v_max, lambda_s, mu_s, rho_s, 
                 mu_f, rho_f, mesh_move, t_end, time_discretization, dt_atol, dt_rtol,
                 result, *args, **kwargs):
        """
        Write boundary conditions, equations and create the files for solution.
        """

        self.E = E
        self.nu_mesh = nu_mesh

        #info("Flow initialization.") 
        self.mesh  = mesh
        self.t     = 0.0
        self.v_max = v_max

        self.mu_f     = mu_f
        self.rho_f    = rho_f
        self.lambda_s = lambda_s
        self.mu_s     = mu_s
        self.rho_s    = rho_s
        
        self.mesh_move = mesh_move
        self.bndry = bndry
        self.interface = interface

        # bounding box tree
        self.bb = BoundingBoxTree()
        self.bb.build(self.mesh)

        # Define finite elements
        eV = VectorElement("CG", mesh.ufl_cell(), 2)		# velocity element 
        eB = VectorElement("Bubble", mesh.ufl_cell(), mesh.geometry().dim()+1) # Bubble element
        eU = VectorElement("CG", mesh.ufl_cell(), 2)		# displacement  element
        eP = FiniteElement("DG", mesh.ufl_cell(), 1)		# pressure element

        eW = MixedElement([eV, eB, eU, eB, eP])			# final mixed element
        W  = FunctionSpace(self.mesh, eW)                       # mixed function space
        self.W = W
        self.V = FunctionSpace(self.mesh, eV)

        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])): \
                      v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),
                      degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        #info("Expression set.")
        bc_v_in     = DirichletBC(self.W.sub(0), self.v_in,            bndry, _INFLOW)
        bc_v_walls  = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_v_circle = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _CIRCLE)
        bc_u_in     = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _INFLOW)
        bc_u_circle = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _CIRCLE)
        #bc_u_walls  = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_u_walls  = DirichletBC(self.W.sub(2).sub(1), Constant(0.0), bndry, _WALLS)
        bc_u_out    = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _OUTFLOW)
        self.bcs = [bc_v_in, bc_v_walls, bc_v_circle, bc_u_in, bc_u_walls, bc_u_circle, bc_u_out]

        #info("Mesh BC.")
        bc_mesh = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), interface, _FSI)
        self.bcs_mesh = [bc_mesh]


        #info("Normal and Circumradius.")
        self.n = FacetNormal(self.mesh)
        self.h = Circumradius(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())

        # Define functions
        self.w  = Function(self.W)      
        self.wdot = Function(self.W)

        (v__, bv_, u__, bu_, p_) = TestFunctions(self.W)
        v_ = v__ + bv_
        u_ = u__ + bu_
        (v, bv, u, bu, self.p) = split(self.w)
        self.v = v + bv
        self.u = u + bu
        (vdot, bvdot, udot, budot, self.pdot) = split(self.wdot)
        self.vdot = vdot + bvdot
        self.udot = udot + budot

        # define deformation gradient, Jacobian
        self.FF  = I + grad(self.u)
        self.JJ  = det(self.FF)

        # write ALE mesh movement 
        self.gamma = 9.0/8.0
        h = CellVolume(self.mesh)**(self.gamma)

        E = Constant(1.0)
        E_mesh = E/h
        nu_mesh = Constant(-0.02)

        mu_mesh = E_mesh/(2*(1.0+nu_mesh))
        lambda_mesh = (nu_mesh*E_mesh)/((1+nu_mesh)*(1-2*nu_mesh))

        F_mesh = inner(mu_mesh*2*sym(grad(self.u)), grad(u_))*dx(0) \
                          + lambda_mesh*inner(div(self.u), div(u_))*dx(0)


        # define referential Grad and Div shortcuts
        def Grad(f): return dot( grad(f), inv(self.FF) )
        def Div(f): return tr( Grad(f) )
        # all dx integrals in the reference should have  self.JJ*dx

        # compute Cauchy stress tensor for fluid
        self.T_f  = -self.p*I  + 2*self.mu_f*sym(Grad(self.v))
        # compute 1st Piola-Kirchhoff tensor for fluid
        self.S_f  = self.JJ *( self.T_f )*inv(self.FF).T

        # write equations for fluid
        a_fluid  = inner(self.T_f , Grad(v_))*self.JJ*dx(0) \
               + inner(self.rho_f*Grad(self.v )*(self.v  - self.udot), v_)*self.JJ*dx(0)

        b_fluid  = inner(Div( self.v ), p_)*self.JJ*dx(0)

        self.F_fluid  = self.rho_f*inner(self.vdot, v_)*self.JJ*dx(0) \
                   + (a_fluid + b_fluid) \
                   + F_mesh

        # compute 1st Piola-Kirchhoff tensor for solid (St. Vennant - Kirchhoff model)
        B_s  = self.FF.T *self.FF
        self.S_s  = self.FF *(0.5*self.lambda_s*tr(B_s  - I)*I + self.mu_s*(B_s  - I))

        # write equation for solid
        alpha = Constant(1.0) # Constant(1e10) #
        self.F_solid = rho_s*inner(self.vdot, v_)*dx(1) \
                   + inner(self.S_s , grad(v_))*dx(1) \
                   + alpha*inner(self.udot - self.v, u_)*dx(1)


        self.dF_fluid_u = derivative(self.F_fluid, self.w)
        self.dF_fluid_udot = derivative(self.F_fluid, self.wdot)
        self.dF_solid_u = derivative(self.F_solid, self.w)
        self.dF_solid_udot = derivative(self.F_solid, self.wdot)

        self.problem = TS.Problem(self.F_fluid, self.F_solid, self.w, self.wdot, 
                  self.bcs_mesh, self.bcs, self.dF_fluid_u, self.dF_fluid_udot, 
                  self.dF_solid_u, self.dF_solid_udot, 
                  update=self.update, report=self.report, form_compiler_parameters=ffc_opts)
        self.solver = TS.Solver(self.problem, tag)

        self.solver.ts.setProblemType(self.solver.ts.ProblemType.NONLINEAR)
        self.solver.ts.setEquationType(self.solver.ts.EquationType.DAE_IMPLICIT_INDEX2)
        if time_discretization == 'BDF':
            self.solver.ts.setType(self.solver.ts.Type.BDF) #BDF #ALPHA #THETA #CN #ARKIMEX #ROSW #BEULER
        elif time_discretization == 'CN':
            self.solver.ts.setType(self.solver.ts.Type.CN) #BDF #ALPHA #THETA #CN #ARKIMEX #ROSW #BEULER
        elif time_discretization == 'BEULER':
            self.solver.ts.setType(self.solver.ts.Type.BEULER) #BDF #ALPHA #THETA #CN #ARKIMEX #ROSW #BEULER
        else:
            raise ValueError('{} is an invalid name for time discretization scheme.'\
                     .format(time_discretiazation))

        #min_step = 1e-02
        min_step = 5e-03			    # BEULER makes long steps when the bounds for time steps are too loose
        self.solver.ts.setTime(0.0)
        self.solver.ts.setMaxTime(t_end)
        self.solver.ts.setTimeStep(min_step)
        self.solver.ts.setMaxSteps(80000)
        self.solver.ts.setMaxStepRejections(-1)
        self.solver.ts.setMaxSNESFailures(-1)       # allow an unlimited number of failures (step will be rejected and retried)
        self.solver.ts.setExactFinalTime(True)

        self.solver.ksp.setType('preonly')
        self.solver.pc.setType('lu')
        #self.solver.pc.setFactorSolverPackage('mumps')		# fenics 2018 (petsc4py 3.8)
        self.solver.pc.setFactorSolverType('mumps')		# fenics 2019 (petsc4py 3.10)
        
        # use adaptive timestep (no petsc4py interface for this)
        opts['ts_adapt_type'] = 'basic' # 'none' # 
        opts['ts_adapt_dt_min'] = 1e-6 # 0.002 #
        opts['ts_adapt_dt_max'] = min_step 

        # set the adaptivity control only on velocity and displacement
        # get the pressure dofs
        pdofs = W.sub(4).dofmap().dofs()

        atol = self.problem.A_petsc.createVecRight()
        rtol = self.problem.A_petsc.createVecRight()
        atol.set(dt_atol)
        rtol.set(dt_rtol)
        atol.setValues(pdofs, [0.0]*len(pdofs))
        rtol.setValues(pdofs, [0.0]*len(pdofs))
        self.solver.ts.setTolerances(atol, rtol)        

        tag.log("system size {0:d}".format(self.W.dim()))
        self.ok=False
        self.its = 0


        # create files for saving
        if my_rank == 0:
            if not os.path.exists(result):
                os.makedirs(result)
        MPI.barrier(comm)
        self.vfile = XDMFFile("%s/velocity.xdmf" % result)
        self.ufile = XDMFFile("%s/displacement.xdmf" % result)
        self.pfile = XDMFFile("%s/pressure.xdmf" % result)
        self.sfile = XDMFFile("%s/stress.xdmf" % result)
        self.vfile.parameters["flush_output"] = True
        self.ufile.parameters["flush_output"] = True
        self.pfile.parameters["flush_output"] = True
        self.sfile.parameters["flush_output"] = True
        with open(result+'/data.csv', 'w') as data_file:
            writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
            writer.writerow(['time', 'mean pressure on outflow', 'pressure_jump', 
                              'x-coordinate of end of beam', 'y-coordinate of end of beam',
                              'pressure difference', 
                              'drag_circle', 'drag_fluid', 'drag_solid', 'drag_fullfluid',
                              'lift_circle', 'lift_fluid', 'lift_solid', 'lift_fullfluid'])

    def solve(self):
        self.its, self.ok = self.solver.solve()
        return(self.its,self.ok)

    def update(self, t):
        self.v_in.t = t

    def report(self, t):
        (v, b1, u, b2, p) = self.w.split()

        v.rename("v", "velocity")
        u.rename("u", "displacement")
        p.rename("p", "pressure")
        self.vfile.write(v, t)
        self.ufile.write(u, t)
        self.pfile.write(p, t)
        P = assemble(self.p*ds(_OUTFLOW))/gW
        PI  = assemble(abs(jump(self.p))*dS(_FSI))

        # Compute drag and lift
        force = dot(self.S_f, self.n)
        D_C = -assemble(force[0]*dss(_FLUID_CYLINDER))
        L_C = -assemble(force[1]*dss(_FLUID_CYLINDER))

        w_ = Function(self.W)
        Fbc = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.interface, _FSI)
        Fbc.apply(w_.vector())
        D_F = -assemble(action(self.F_fluid,w_))
        w_ = Function(self.W)
        Fbc = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.interface, _FSI)
        Fbc.apply(w_.vector())        
        L_F = -assemble(action(self.F_fluid,w_))

        w_ = Function(self.W)
        Fbc = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.interface, _FSI)
        Fbc.apply(w_.vector())
        D_S = assemble(action(self.F_solid,w_))
        w_ = Function(self.W)
        Fbc = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.interface, _FSI)
        Fbc.apply(w_.vector())        
        L_S = assemble(action(self.F_solid,w_))

        w_ = Function(self.W)
        Fbc1 = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.interface, _FLUID_CYLINDER)
        Fbc2 = DirichletBC(self.W.sub(0), Constant((1.0, 0.0)), self.interface, _FSI)
        Fbc1.apply(w_.vector())
        Fbc2.apply(w_.vector())
        D_FF = -assemble(action(self.F_fluid,w_))
        w_ = Function(self.W)
        Fbc1 = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.interface, _FLUID_CYLINDER)
        Fbc2 = DirichletBC(self.W.sub(0), Constant((0.0, 1.0)), self.interface, _FSI)
        Fbc1.apply(w_.vector())
        Fbc2.apply(w_.vector())
        L_FF = -assemble(action(self.F_fluid,w_))


        # MPI trick to extract displacement values at the end of the beam
        self.w.set_allow_extrapolation(True)
        pA_loc = self.p((A.x(), A.y()))
        pB_loc = self.p((B.x(), B.y()))
        pB_loc = self.p((B.x(), B.y()))
        Ax_loc = self.u[0]((A.x(), A.y()))
        Ay_loc = self.u[1]((A.x(), A.y()))
        self.w.set_allow_extrapolation(False)
        
        pi = 0
        if self.bb.compute_first_collision(A) < 4294967295:
            pi = 1
        else:
            pA_loc = 0.0
            Ax_loc = 0.0
            Ay_loc = 0.0
        pA = MPI.sum(comm, pA_loc) / MPI.sum(comm, pi)
        Ax = MPI.sum(comm, Ax_loc) / MPI.sum(comm, pi)
        Ay = MPI.sum(comm, Ay_loc) / MPI.sum(comm, pi)

        pi = 0
        if self.bb.compute_first_collision(B) < 4294967295:
            pi = 1
        else:
            pB_loc = 0.0
        pB = MPI.sum(comm, pB_loc) / MPI.sum(comm, pi)
        p_diff = pB - pA

        # wirte computed quantities to a file
        if my_rank == 0:
            with open(result+'/data.csv', 'a') as data_file:
                writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
                writer.writerow([t, P, PI, Ax, Ay, p_diff, D_C, D_F, D_S, D_FF, L_C, L_F, L_S, L_FF])


def get_benchmark_specification(benchmark = 'FSI1'):
    """
    routine for obtaining benchmark-specific constants.
    """
    if benchmark == 'FSI1':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 0.2
        T_end = 60.0
        result = "results-FSI1/"
    elif benchmark == 'FSI2':
        rho_s = Constant(1e04)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 1.0
        T_end = 15.0
        result = "results-FSI2/"		
    elif benchmark == 'FSI3':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(2e06)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 2.0
        T_end = 20.0
        result = "results-FSI3/"		
    else:
        raise ValueError('"{}" is a wrong name for problem specification.'.format(benchmark))
    v_max = Constant(1.5*U)     # mean velocity to maximum velocity 
                                #      (we have parabolic profile)
    E_s = Constant(2*mu_s*(1+nu_s))
    lambda_s = Constant((nu_s*E_s)/((1+nu_s)*(1-2*nu_s)))
    mu_f = Constant(nu_f*rho_f)
    return v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, T_end, result


# set problem and its discretization
parser = OptionParser()
parser.add_option("--benchmark", dest="benchmark", default='FSI2')
parser.add_option("--mesh", dest="mesh_name", default='mesh_ALE_L1')
parser.add_option("--time_discretization", dest="time_discretization", default='BDF')
parser.add_option("--tol", dest="tol", default='a104r104')

(options, args) = parser.parse_args()

# name of benchmark 
benchmark = options.benchmark

# name of mesh
mesh_name = options.mesh_name
relative_path_to_mesh = '../meshes/'+mesh_name+'.h5'

# time discretization scheme
time_discretization = options.time_discretization

# tolerances to addaptive time stepping
tolerances = options.tol

if tolerances == 'a104r104':
    atol = 1.0e-04
    rtol = 1.0e-04
elif tolerances == 'a104r105':
    atol = 1.0e-04
    rtol = 1.0e-05
elif tolerances == 'a105r105':
    atol = 1.0e-05
    rtol = 1.0e-05
else:
    raise ValueError('Invalid value for flag --tol.')

v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, t_end, result = get_benchmark_specification(benchmark)
result = result + tolerances + '/'+ time_discretization + '/' + mesh_name[:-3] + '/' + mesh_name[-2:]

# load mesh with boundary and domain markers
sys.path.append('../meshes/')
import marker

#(mesh, bndry, domains, interface, A, B) \
#        = marker.give_marked_mesh(mesh_coarseness = mesh_coarseness, refinement = True, ALE = True)
(mesh, bndry, domains, interface, A, B) = \
         marker.give_gmsh_mesh(relative_path_to_mesh)

# domain (used while building mesh) - needed for inflow condition
gW = 0.41

# boundary marks' names (already setted to the mesh) - needed for boundary conditions
_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

# interface marks
_FSI = 1
_FLUID_CYLINDER = 2


dx  = dx(domain=mesh, subdomain_data = domains)
ds  = ds(domain=mesh, subdomain_data = bndry)
dss = ds(domain=mesh, subdomain_data = interface)
dS  = dS(domain=mesh, subdomain_data = interface)

################# export domains and bndry to xdmf for visualization in Paraview
# facet markers can be written directly
with XDMFFile("%s/mesh_bndry.xdmf" % result) as f:
    f.write(bndry)

with XDMFFile("%s/mesh_interface.xdmf" % result) as f:
    f.write(interface)

# subdomain markers has to be firs interpolated to DG0 space
class DomainsProjection(UserExpression):
    def __init__(self, mesh, domains, **kwargs):
        self.mesh = mesh
        self.domains=domains
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        values[0] = self.domains[cell]
    def value_shape(self):
        return ()
    
with XDMFFile("%s/mesh_domains.xdmf" % result) as f:
    dde = DomainsProjection(mesh, domains, degree=0)
    ddf = interpolate(dde, FunctionSpace(mesh, "DG", 0))
    f.write(ddf)
############################################################################

import builtins
builtins.tag=Report(2.5)
tag.time()

tag.begin('Total_ALE_PETSc')


if time_discretization == 'BDF_CN':
    flow = Flow(mesh, bndry, interface, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, 
             mesh_move, t_end, 'BDF', atol, rtol, E, nu_mesh, result)
    flow.solver.ts.setTime(0.0)
    flow.solver.ts.setMaxTime(2.0)
    its, ok = flow.solve()
    tag.end()

    tag.begin('t>2.0')
    flow.solver.ts.setType(flow.solver.ts.Type.CN) #BDF #ALPHA #THETA #CN #ARKIMEX #ROSW #BEULER
    flow.solver.ts.setTime(2.0)
    flow.solver.ts.setMaxTime(t_end)
else:
    flow = Flow(mesh, bndry, interface, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, 
             t_end, time_discretization, atol, rtol, result)


its, ok = flow.solve()
tag.end()

#with open('convergence.log', 'a') as logfile:
#    if my_rank == 0:
#        logfile.write('E = {}, nu_mesh = {}, gamma = {} \t succesful! \t number of time steps = {}\n'.format(
#               E, nu_mesh, flow.gamma, flow.solver.ts.getStepNumber()))     
