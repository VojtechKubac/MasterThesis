"""
This code was implemented by Vojtech Kubac as a part of his Master Thesis that will be defended
on February 2020. At the Faculty of Mathematics and Physics of Charles University in Prague.
"""

"""
This code solves FSI2 and FSI3 Bencmarks from
    "S. Turek and J. Hron, “Proposal for numerical benchmarking of fluid–
     structure interaction between an elastic object and laminar incompress-
     ible flow,” in Fluid-Structure Interaction - Modelling, Simulation, Opti-
     mization, ser. Lecture Notes in Computational Science and Engineering,"

other FSI simulations can be run by loading corresponding mesh and a straightforward 
modifications of boundary conditions.

The equations are written in Total-ALE formulation, where for the mesh movement pseudoelasticity
extension of the solid displacement was used.

Chosen Finite Elements are linear discontinuous space for pressure and quadratic continuous space
enriched with quadratic bubble is used for displacement and velocity.

Time discretization scheme is theta-scheme with default value 0.5, which means Crank-Nicolson.
The equation for pressure and incompressibility equation are discretized by implicit Euler.
"""

from dolfin import *
from dolfin import __version__
import mshr
import numpy as np
import csv
import sys
import os.path
from mpi4py.MPI import COMM_WORLD
from optparse import OptionParser


# Define MPI World
if __version__[:4] == '2017':
    comm = mpi_comm_world()
else:
    comm = MPI.comm_world
my_rank = comm.Get_rank()

# Use UFLACS to speed-up assembly and limit quadrature degree
parameters["std_out_all_processes"] = False
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4

parameters['ghost_mode'] = 'shared_facet'

PETScOptions.set('mat_mumps_icntl_24', 1)		# detects null pivots
PETScOptions.set('mat_mumps_cntl_1', 0.01)		# set treshold for partial treshold pivoting, 0.01 is default value

class Problem(NonlinearProblem):
    """
    Nonlinear problem for solving System of nonlinear equations that arises from 
    Finite Elemnt discretization of the equtions describing the FSI phenomenon.

    It inherites methods from FEniCS class NonlinearProblem. But redefines methods 'F' and 'J'
    in such a way that it nulls elements corresponding to the artificial mesh-moving equation
    on the interface with elastic solid. This guarantees that the mesh-moving ALE equation for fluid
    does not influence the solution to elasticity displacement.
    """
    def __init__(self, F_mesh, FF, dF_mesh, dF, bcs_mesh, bcs):
        NonlinearProblem.__init__(self)
        self.F_mesh = F_mesh
        self.FF = FF
        self.dF_mesh = dF_mesh
        self.dF = dF
        self.bcs_mesh = bcs_mesh
        self.bcs = bcs
        self.assembler = SystemAssembler(dF+dF_mesh, FF+F_mesh, bcs+bcs_mesh) 
        self.A1 = PETScMatrix()
        self.A2 = PETScMatrix()

    def F(self, b, x):
        self.assembler.init_global_tensor(b, Form(self.FF+self.F_mesh))
        b.apply('add')
        
        b1=Vector()
        b2=Vector()
        assemble(self.F_mesh, tensor = b1)
        [bc.apply(b1) for bc in self.bcs_mesh]

        assemble(self.FF, tensor = b2)
                                                        
        b.axpy(1,b1)
        b.axpy(1,b2)
        [bc.apply(b, x) for bc in self.bcs]
        
    def J(self, A, x):
        self.assembler.init_global_tensor(A, Form(self.dF+self.dF_mesh))
        A.apply('insert')

        assemble(self.dF_mesh, tensor = self.A1, keep_diagonal=True)
        [bc.zero(self.A1) for bc in self.bcs_mesh]
        assemble(self.dF, tensor = self.A2, keep_diagonal=True)

        A.axpy(1, self.A1, False)
        A.axpy(1, self.A2, False)
        [bc.apply(A) for bc in self.bcs]


class Flow(object):
    """
    Class where the equations for the FSI are defined. It possesses methods 'solve' and 'save'
    that solves equations in each time step and  then saves the obtained results.
    """
    def __init__(self, mesh, bndry, interface, dt, theta, v_max, lambda_s, mu_s, rho_s, 
                 mu_f, rho_f, result, *args, **kwargs):
        """
        Write boundary conditions, equations and create the files for solution.
        """

        self.mesh  = mesh
        self.dt    = Constant(dt)
        self.theta = theta
        self.t     = 0.0
        self.v_max = v_max

        self.mu_f     = mu_f
        self.rho_f    = rho_f
        self.lambda_s = lambda_s
        self.mu_s     = mu_s
        self.rho_s    = rho_s
        
        self.bndry = bndry
        self.interface = interface

        # bounding box tree
        self.bb = BoundingBoxTree()
        self.bb.build(self.mesh)

        # Define finite elements
        eV = VectorElement("CG", mesh.ufl_cell(), 2)		# velocity element
        eB = VectorElement("Bubble", mesh.ufl_cell(), mesh.geometry().dim()+1) # Bubble element
        eU = VectorElement("CG", mesh.ufl_cell(), 2)		# displacement element
        eP = FiniteElement("DG", mesh.ufl_cell(), 1)		# pressure element

        eW = MixedElement([eV, eB, eU, eB, eP])                 # final mixed element
        W  = FunctionSpace(self.mesh, eW)                       # mixed space
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
        bc_u_walls  = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _WALLS)
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
        self.w  = Function(self.W)      # solution to current time step
        self.w0 = Function(self.W)      # solution from previous time step

        (v__, bv_, u__, bu_, p_) = TestFunctions(self.W)

        # sum bubble elements with corresponding Lagrange elements
        v_ = v__ + bv_
        u_ = u__ + bu_
        (v, bv, u, bu, self.p) = split(self.w)
        self.v = v + bv
        self.u = u + bu
        (v0, bv0, u0, bu0, self.p0) = split(self.w0)
        self.v0 = v0 + bv0
        self.u0 = u0 + bu0


        # define deformation gradient, Jacobian
        self.FF  = I + grad(self.u)
        self.FF0 = I + grad(self.u0)
        self.JJ  = det(self.FF)
        self.JJ0 = det(self.FF0)

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
        def Grad(f, F): return dot( grad(f), inv(F) )
        def Div(f, F): return tr( Grad(f, F) )

        # approximate time derivatives
        du = (1.0/self.dt)*(self.u - self.u0)
        dv = (1.0/self.dt)*(self.v - self.v0)

        # compute velocuty part of Cauchy stress tensor for fluid
        self.T_f  = -self.p*I + 2*self.mu_f*sym(Grad(self.v,  self.FF))
        self.T_f0 = -self.p*I + 2*self.mu_f*sym(Grad(self.v0, self.FF0))

        # Compute 1st Piola-Kirhhoff tensro for fluid 
        #       - for computing surface integrals for forces in postprocessing 
        self.S_f  = self.JJ *self.T_f*inv(self.FF).T
        
        # write equations for fluid
        a_fluid  = inner(self.T_f , Grad(v_, self.FF))*self.JJ*dx(0) \
               - inner(self.p, Div(v_, self.FF))*self.JJ*dx(0) \
               + inner(self.rho_f*Grad(self.v, self.FF )*(self.v  - du), v_)*self.JJ*dx(0)
        a_fluid0 = inner(self.T_f0, Grad(v_, self.FF0))*self.JJ0*dx(0) \
               - inner(self.p, Div(v_, self.FF))*self.JJ*dx(0) \
               + inner(self.rho_f*Grad(self.v0, self.FF0)*(self.v0 - du), v_)*self.JJ0*dx(0)

        b_fluid  = inner(Div( self.v, self.FF ), p_)*self.JJ*dx(0)
        b_fluid0 = inner(Div( self.v, self.FF ), p_)*self.JJ*dx(0)

        self.F_fluid  = (self.theta*self.JJ+(1.0 - self.theta)*self.JJ0)*self.rho_f*inner(dv, v_)*dx(0)\
                   + self.theta*(a_fluid + b_fluid) + (1.0 - self.theta)*(a_fluid0 + b_fluid0) \
                   + F_mesh

        # compute 1st Piola-Kirchhoff tensor for solid (St. Vennant - Kirchhoff model)
        B_s  = self.FF.T *self.FF
        B_s0 = self.FF0.T*self.FF0
        S_s  = self.FF *(0.5*self.lambda_s*tr(B_s  - I)*I + self.mu_s*(B_s  - I))
        S_s0 = self.FF0*(0.5*self.lambda_s*tr(B_s0 - I)*I + self.mu_s*(B_s0 - I))

        # write equation for solid
        alpha = Constant(1.0) # Constant(1e10) #
        self.F_solid = rho_s*inner(dv, v_)*dx(1) \
                   + self.theta*inner(S_s , grad(v_))*dx(1) + (1.0 - self.theta)*inner(S_s0, grad(v_))*dx(1) \
                   + alpha*inner(du - (self.theta*self.v + (1.0 - self.theta)*self.v0), u_)*dx(1)


        dF_solid = derivative(self.F_solid, self.w)
        dF_fluid = derivative(self.F_fluid, self.w)

        self.problem = Problem(self.F_fluid, self.F_solid, dF_fluid, dF_solid, self.bcs_mesh, self.bcs)
        self.solver = NewtonSolver()

        # configure solver parameters
        self.solver.parameters['relative_tolerance'] = 1e-6
        self.solver.parameters['maximum_iterations'] = 15
        self.solver.parameters['linear_solver']      = 'mumps'

        # create files for saving
        if my_rank == 0:
            if not os.path.exists(result):
                os.makedirs(result)
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

    def solve(self, t, dt):
        self.t = t
        self.v_in.t = t
        self.dt = Constant(dt)
        self.solver.solve(self.problem, self.w.vector())

        self.w0.assign(self.w)

    def save(self, t):
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


        # MPI trick to extract displacement of the end of the beam
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

        # write computed quantities to a csv file
        if my_rank == 0:
            with open(result+'/data.csv', 'a') as data_file:
                writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
                writer.writerow([t, P, PI, Ax, Ay, p_diff, D_C, D_F, D_S, D_FF, L_C, L_F, L_S, L_FF])



def get_benchmark_specification(benchmark = 'FSI1'):
    """
    Method for obtaining the right problem-specific constants.
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
parser.add_option("--dt", dest="dt", default='0.001')
parser.add_option("--dt_scheme", dest="dt_scheme", default='CN')	# BE BE_CN

(options, args) = parser.parse_args()

# name of benchmark 
benchmark = options.benchmark

# name of mesh
mesh_name = options.mesh_name
relative_path_to_mesh = 'meshes/'+mesh_name+'.h5'

# time step size
dt = options.dt

# time stepping scheme
dt_scheme = options.dt_scheme

# choose theta according to dt_scheme
if dt_scheme in ['BE', 'BE_CN']:
    theta = Constant(1.0)
elif dt_scheme == 'CN':
    theta = Constant(0.5)
else:
    raise ValueError('Invalid argument for dt_scheme')

v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, t_end, result = get_benchmark_specification(benchmark)
result = result + 'dt_' + str(dt) + '/' + dt_scheme + '/' + mesh_name[:-3] + '/' + mesh_name[-2:]

# load mesh with boundary and domain markers
sys.path.append('../meshes')
import marker

#(mesh, bndry, domains, interface, A, B) \
#        = marker.give_marked_mesh(mesh_coarseness = mesh_coarseness, refinement = True, ALE = True)
(mesh, bndry, domains, interface, A, B) = marker.give_gmsh_mesh(relative_path_to_mesh)

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

flow = Flow(mesh, bndry, interface, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result)

t = 0.0

while t < 2.0:
    if my_rank == 0: 
        info("t = %.4f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    flow.save(t)

    t += float(dt)

if dt_scheme == 'BE_CN': flow.theta.assign(0.5)

while  t < t_end:
    if my_rank == 0: 
        info("t = %.4f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    flow.save(t)

    t += float(dt)
