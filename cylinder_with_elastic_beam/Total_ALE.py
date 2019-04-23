from dolfin import *
from dolfin import __version__
import mshr
import numpy as np
import csv
import sys
import os.path
from mpi4py.MPI import COMM_WORLD


if __version__[:4] == '2018':
    comm = MPI.comm_world
else:
    comm = mpi_comm_world()
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
    def __init__(self, mesh, bndry, interface, dt, theta, v_max, lambda_s, mu_s, rho_s, 
                 mu_f, rho_f, mesh_move, result, *args, **kwargs):

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
        
        self.mesh_move = mesh_move
        self.bndry = bndry
        self.interface = interface

        # bounding box tree
        self.bb = BoundingBoxTree()
        self.bb.build(self.mesh)

        # Define finite elements
        eV = VectorElement("CG",     mesh.ufl_cell(), 2)	# velocity element
        eB = VectorElement("Bubble", mesh.ufl_cell(), 3)        # Bubble element
        eU = VectorElement("CG",     mesh.ufl_cell(), 2)	# displacement element
        eP = FiniteElement("DG",      mesh.ufl_cell(), 1)       # pressure element
        eW = MixedElement([eV, eB, eU, eB, eP])			# function space
        W  = FunctionSpace(self.mesh, eW)
        self.W = W
        self.V = FunctionSpace(self.mesh, eV)

        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])): \
                      v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),
                      degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        bc_v_in     = DirichletBC(self.W.sub(0), self.v_in,            bndry, _INFLOW)
        bc_v_walls  = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_v_circle = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _CIRCLE)
        bc_u_in     = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _INFLOW)
        bc_u_circle = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _CIRCLE)
        bc_u_walls  = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_u_out    = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), bndry, _OUTFLOW)
        self.bcs = [bc_v_in, bc_v_walls, bc_v_circle, bc_u_in, bc_u_walls, bc_u_circle, bc_u_out]

        bc_mesh = DirichletBC(self.W.sub(2), Constant((0.0, 0.0)), interface, _FSI)
        self.bcs_mesh = [bc_mesh]


        self.n = FacetNormal(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())

        # Define functions
        self.w  = Function(self.W)      
        self.w0 = Function(self.W)

        (v__, bv_, u__, bu_, p_) = TestFunctions(self.W)
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
        E_mesh = Expression("( pow(x[0] - 0.425, 2) < 0.04 ? sqrt(pow(x[1] - 0.2, 2)) : \
                    pow(x[1] - 0.2, 2) < 0.0001 ? sqrt(pow(x[0] - 0.425, 2)): \
                    sqrt(pow(x[0] - 0.425, 2) + pow(x[1] - 0.2, 2)) ) < 0.06 ? \
                    1e06 : 8e04", degree=1)

        nu_mesh = Constant(-0.1)
        mu_mesh = E_mesh/(2*(1.0+nu_mesh))
        lambda_mesh = (nu_mesh*E_mesh)/((1+nu_mesh)*(1-2*nu_mesh))

        F_mesh = inner(mu_mesh*2*sym(grad(self.u)), grad(u_))*dx(0) \
                          + lambda_mesh*inner(div(self.u), div(u_))*dx(0)

        # approximate time derivatives
        du = (1.0/self.dt)*(self.u - self.u0)
        dv = (1.0/self.dt)*(self.v - self.v0)

        # compute 1st Piola-Kirchhoff tensor for fluid
        self.S_f  = self.JJ *( -self.p*I  + 2*self.mu_f*sym(grad(self.v )) )*inv(self.FF).T
        self.S_f0 = self.JJ0*( 2*self.mu_f*sym(grad(self.v0)) )*inv(self.FF0).T \
                     - self.JJ *(self.p*I)*inv(self.FF).T

        # write equations for fluid
        a_fluid  = inner(self.S_f , grad(v_))*dx(0) \
                   + inner(self.JJ *self.rho_f*grad(self.v )*inv(self.FF ).T*(self.v  - du), v_)*dx(0)
        a_fluid0 = inner(self.S_f0, grad(v_))*dx(0) \
                   + inner(self.JJ0*self.rho_f*grad(self.v0)*inv(self.FF0).T*(self.v0 - du), v_)*dx(0)

        b_fluid  = inner(div(self.JJ *inv(self.FF )*self.v ), p_)*dx(0)
        b_fluid0 = inner(div(self.JJ *inv(self.FF )*self.v ), p_)*dx(0)

        self.F_fluid  = self.JJ*self.rho_f*inner(dv, v_)*dx(0) \
                   + self.theta*(a_fluid + b_fluid) + (1.0 - self.theta)*(a_fluid0 + b_fluid0) \
                   + F_mesh

        # compute 1st Piola-Kirchhoff tensor for solid (St. Vennant - Kirchhoff model)
        B_s  = self.FF.T *self.FF
        B_s0 = self.FF0.T*self.FF0
        S_s  = self.FF *(0.5*self.lambda_s*tr(B_s  - I)*I + self.mu_s*(B_s  - I))
        S_s0 = self.FF0*(0.5*self.lambda_s*tr(B_s0 - I)*I + self.mu_s*(B_s0 - I))

        # write equation for solid
        self.F_solid = rho_s*inner(dv, v_)*dx(1) \
                   + self.theta*inner(S_s , grad(v_))*dx(1) \
                   + (1.0 - self.theta)*inner(S_s0, grad(v_))*dx(1) \
                   + inner(du - (self.theta*self.v + (1.0 - self.theta)*self.v0), u_)*dx(1)

        # write final equation
        F = self.F_solid

        dF = derivative(F, self.w)
        dF_fluid= derivative(self.F_fluid, self.w)

        self.problem = Problem(self.F_fluid, F, dF_fluid, dF, self.bcs_mesh, self.bcs)
        self.solver = NewtonSolver()

        # configure solver parameters
        self.solver.parameters['relative_tolerance'] = 1e-6
        self.solver.parameters['maximum_iterations'] = 15
        self.solver.parameters['linear_solver']      = 'mumps'

        # create files for saving
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
        self.data = open(result+'/data.csv', 'w')
        self.writer = csv.writer(self.data, delimiter=';', lineterminator='\n')
        self.writer.writerow(['time', 'mean pressure on outflow', 'pressure difference', 
                              'x-coordinate of end of beam', 'y-coordinate of end of beam',
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
        v = project(v + b1, self.V)
        u = project(u + b2, self.V)

        v.rename("v", "velocity")
        u.rename("u", "displacement")
        p.rename("p", "pressure")
        self.vfile.write(v, t)
        self.ufile.write(u, t)
        self.pfile.write(p, t)
        P = assemble(self.p*ds(_OUTFLOW))/gW

        # Compute drag and lift
        force = dot(self.S_f, self.n)
        D_C = -assemble(force[0]*dss(1))
        L_C = -assemble(force[1]*dss(1))

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


        #info("Extracting values")
        self.w.set_allow_extrapolation(True)
        pA_loc = self.p((A.x(), A.y()))
        pB_loc = self.p((B.x(), B.y()))
        pB_loc = self.p((B.x(), B.y()))
        Ax_loc = self.u[0]((A.x(), A.y()))
        Ay_loc = self.u[1]((A.x(), A.y()))
        self.w.set_allow_extrapolation(False)
        
        #info("collision for A.")
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

        self.writer.writerow([t, P, p_diff, Ax, Ay, D_C, D_F, D_S, D_FF, L_C, L_F, L_S, L_FF])

    def get_integrals(self):
        self.area        = assemble(1*dx)
        area_fluid       = assemble(1*dx(0))
        area_solid       = assemble(1*dx(1))
        elastic_surface  = assemble(1*dS(_FSI))
        solid_surface    = elastic_surface + assemble(1*dss(_FLUID_CYLINDER))

        if my_rank == 0:
            info("area of the whole domain = {}, \narea of fluid domain = {},".format(self.area,
                                                                                   area_fluid ))
            info("area of solid domain = {}, \nelastic surface = {}, \nsolid surface = {}".format(
                                        area_solid, elastic_surface, solid_surface))
            info("Degrees of freedom = {}".format(self.W.dim()))
                 

if len(sys.argv) > 1:
    benchmark = str(sys.argv[1])
else:
    benchmark = 'FSI1'

# load mesh with boundary and domain markers
sys.path.append('.')
import utils 

(mesh, bndry, domains, interface, A, B) = utils.give_gmsh_mesh('meshes/mesh2.h5')

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

v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, t_end, dt, result \
        = utils.get_benchmark_specification(benchmark)
result = 'Total_ALE_' + result + '_'
theta = Constant(0.5)

flow = Flow(mesh, bndry, interface, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, 
             mesh_move, result)

flow.get_integrals()

t = 0.0


flow.theta.assign(1.0)
if benchmark == 'FSI3':
    dt = 0.0002
    while  t < 0.001:
        if my_rank == 0: 
            info("t = %.4f, t_end = %.1f" % (t, t_end))
        flow.solve(t, dt)
        flow.save(t)

        t += float(dt)
    dt = 0.0005

    

while  t < 2.0:
    if my_rank == 0: 
        info("t = %.4f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    flow.save(t)

    t += float(dt)

if benchmark == 'FSI2':
    flow.theta.assign(0.5)
while  t < t_end:
    if my_rank == 0: 
        info("t = %.4f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    flow.save(t)

    t += float(dt)

flow.data.close()
