"""
This code was implemented by Vojtech Kubac as a part of his Master Thesis that will be defended
on February 2020. At the Faculty of Mathematics and Physics of Charles University in Prague.
"""

"""
Fully Eulerian formulation of Turek&Hron FSI benchmark,
cut cell integration performed with the use of integration_cutcell library
, stored in integration_cutcell.py
"""

from dolfin import *
from dolfin import __version__
from dolfin.fem.assembling import _create_dolfin_form
import numpy as np
import dijitso
import FIAT
import time
import sys
import csv
sys.path.append('.')
from integration_cutcell import assemble_cutcell, assemble_cutcell_time_dependent


parameters["std_out_all_processes"] = True

# Use UFLACS to speed-up assembly and limit quadrature degree
#parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4

parameters['ghost_mode'] = 'shared_vertex'

PETScOptions.set('mat_mumps_icntl_24', 1)	# detects null pivots
PETScOptions.set('mat_mumps_cntl_1', 0.01)	# set treshold for partial treshold pivoting, 0.01 is default value

if __version__[:4] == '2017':
    comm = mpi_comm_world()
else:
    comm = MPI.comm_world
my_rank = comm.Get_rank()


class Problem(NonlinearProblem):
    def __init__(self, F_fluid, F_fluid0, F_fluid_cutcell, F_fluid_cutcell0, 
                     F_solid, F_solid0, F_solid_cutcell, F_solid_cutcell0,
                     dF_fluid, dF_fluid_cutcell, dF_solid, dF_solid_cutcell, 
                     w, bcs, cutcells, V, ips, ips0):
        #info("Initialize NonlinearProblem.")
        NonlinearProblem.__init__(self)
        self.F_fluid = F_fluid
        self.F_fluid0 = F_fluid0
        self.F_fluid_cutcell = _create_dolfin_form(F_fluid_cutcell*dC)
        self.F_fluid_cutcell0 = _create_dolfin_form(F_fluid_cutcell0*dC)
        self.F_solid = Form(F_solid - F_fluid)
        self.F_solid0 = Form(F_solid0 - F_fluid0)
        self.F_solid_cutcell = _create_dolfin_form(F_solid_cutcell*dC)
        self.F_solid_cutcell0 = _create_dolfin_form(F_solid_cutcell0*dC)
        self.dF_fluid = Form(dF_fluid)
        self.dF_fluid_cutcell = _create_dolfin_form(dF_fluid_cutcell*dC)
        self.dF_solid = Form(dF_solid - dF_fluid)
        self.dF_solid_cutcell = _create_dolfin_form(dF_solid_cutcell*dC)
        self.bcs = bcs
        self.cutcells = cutcells
        #self.newcutcells = cutcells
        self.V = V
        self.ips_ = ips
        self.ips0_ = ips0


        #N = Expression(("a","b"), degree=0, a=1, b=1)
        self.F_solid_interface_x = _create_dolfin_form(w[0]*F_solid_cutcell*dC)
        self.F_fluid_interface_x = _create_dolfin_form(w[0]*F_fluid_cutcell*dC)
        self.F_solid_interface_y = _create_dolfin_form(w[1]*F_solid_cutcell*dC)
        self.F_fluid_interface_y = _create_dolfin_form(w[1]*F_fluid_cutcell*dC)

        u = Function(V)
        u_ = TestFunction(V)
        u_trial = TrialFunction(V)
        self.zero_form = Constant(0)*inner(u, u_)*dx
        self.zero_form = Form(self.zero_form)
        self.zero_trial_form = Constant(0)*inner(u_trial, u_)*dx
        self.zero_trial_form = Form(self.zero_trial_form)

        #self.assembler = SystemAssembler(dF_fluid+dF_solid, F_fluid+F_solid, bcs+bcs_mesh) 
        #self.A1 = PETScMatrix()
        #self.A2 = PETScMatrix()

    def F(self, b, x):
        #self.assembler.init_global_tensor(b, Form(self.F_fluid+self.F_solid))
        #b.apply('add')

        #b1=Vector()
        #b2=Vector()
        assemble(self.F_fluid + self.F_fluid0, tensor=b)#, finalize=False)
             # kvuli scitani forem musim asemblovat az pri volani 'assemble_cutcell_time_dependent'

        #info('Call assemble_cutcell_time_dependent')
        #self.newcutcells.clear()
        #self.newcutcells = assemble_cutcell_time_dependent(b,Form(self.F_fluid),Form(self.F_fluid0), 
        assemble_cutcell_time_dependent(b,Form(self.F_fluid),Form(self.F_fluid0), self.F_fluid_cutcell, 
                self.F_fluid_cutcell0, self.F_solid, self.F_solid0, self.F_solid_cutcell, 
                self.F_solid_cutcell0, self.zero_form, self.cutcells, self.V, self.ips_, self.ips0_)

        [bc.apply(b, x) for bc in self.bcs]

    def J(self, A, x):
        #self.assembler.init_global_tensor(A, Form(self.dF+self.dF_mesh))
        #A.apply('insert')

        assemble(self.dF_fluid, tensor = A, keep_diagonal=True)#, finalize_tensor=False, add_values=True)

        #info('Call assemble_cutcell')
        assemble_cutcell(A, self.dF_fluid, self.dF_fluid_cutcell, self.dF_solid, self.dF_solid_cutcell,
                              self.zero_trial_form, self.F_fluid_interface_x, self.F_solid_interface_x, 
                              self.F_fluid_interface_y, self.F_solid_interface_y,
                              self.cutcells, self.V, self.ips_)


        #A.axpy(1, self.A1, False)
        #A.axpy(1, self.A2, False)
        [bc.apply(A) for bc in self.bcs]




class Flow(object):
    def __init__(self, mesh, bndry, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result, 
            *args, **kwargs):

        result = result

        self.mesh  = mesh
        self.bndry = bndry
        self.dt    = dt
        self.theta = theta
        self.t     = 0.0
        self.v_max = v_max

        self.mu_f     = mu_f
        self.rho_f    = rho_f
        self.lambda_s = lambda_s
        self.mu_s     = mu_s
        self.rho_s    = rho_s

        # Define finite elements
        eV = VectorElement("CG", mesh.ufl_cell(), 2)
        eU = VectorElement("CG", mesh.ufl_cell(), 2)
        eP = FiniteElement("CG", mesh.ufl_cell(), 1)
        eW = MixedElement([eV, eU, eP])
        W  = FunctionSpace(self.mesh, eW)
        self.W = W
        self.U = FunctionSpace(self.mesh, eU)
        self.P = FunctionSpace(self.mesh, eP)
    
        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])): \
                   v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),\
                  degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        bc_u_circle = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _CIRCLE)
        bc_v_in     = DirichletBC(self.W.sub(0), self.v_in,            bndry, _INFLOW)
        bc_v_walls  = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_v_circle = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _CIRCLE)

        self.bcs = [bc_v_in, bc_v_walls, bc_v_circle, bc_u_circle]

        self.potentially_cutcells = set() # to avoid having one cell multiple times
        """
        proces vidi jenom ty bunky, ktere mu prisluseji, tedy by se nemelo stat, ze
        budu integrovat pres bunku, kterou nevlastni dany proces
        """
        for c in cells(mesh):
            mp = c.midpoint()
            if (mp[0] <= 0.64 and mp[0] >= 0.2 and mp[1] <= 0.3 and mp[1] >= 0.1):
            #if domains[c] == 1:
                self.potentially_cutcells.add(c)
        #self.potentially_cutcells = set() # to avoid having one cell multiple times
        #info('{}'.format(self.potentially_cutcells))
        

        self.n = FacetNormal(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())

        # Define test functions
        (v_, u_, p_) = TestFunctions(self.W)

        # current unknown time step
        self.w = Function(self.W)
        (self.v, self.u, self.p) = split(self.w)

        # previous known time step
        self.w0 = Function(self.W)
        (self.v0, self.u0, self.p0) = split(self.w0)

        # definition of Initial Point Set function
        #       - the displacement inside IPS needs to be updated at every time step
        parameters["allow_extrapolation"] = True
        self.w.set_allow_extrapolation(True)
        self.w0.set_allow_extrapolation(True)

        self.ips = Expression("(x[0]-u0 > gLs1 && x[0]-u0 < gLs2 && x[1]-u1 > gWs1 && x[1]-u1 <gWs2)? \
                1 : 0", 
                u0 = (project(self.u, self.U)).sub(0), u1 = (project(self.u, self.U)).sub(1), 
                gLs1 = gLs1, gLs2 = gLs2, gWs1 = gWs1, gWs2 = gWs2, degree = 1)

        self.ips0 = Expression("(x[0]-u0 > gLs1 && x[0]-u0 < gLs2 && x[1]-u1 > gWs1 && x[1]-u1 <gWs2)? \
                1 : 0", 
                u0 = (project(self.u0, self.U)).sub(0), u1 = (project(self.u0, self.U)).sub(1), 
                gLs1 = gLs1, gLs2 = gLs2, gWs1 = gWs1, gWs2 = gWs2, degree = 1)

        self.ips((0.0, 0.0))
        self.ips0((0.0, 0.0)) 

        # trial function for derivatives
        Dw = TrialFunction(self.W)
        #(Dv, Dvb, Du, Dub, Dp) = Dw.split()
        (Dv, Du, Dp) = split(Dw)
        #Dv = Dv + Dvb
        #Du = Du + Dub

        # define deformation gradient and Jacobian
        #     (from current configuration to initial)
        self.FF  = I - grad(self.u)
        self.FF0 = I - grad(self.u0)
        JJ  = det(self.FF)
        JJ0 = det(self.FF0)

        # approximate time derivatives
        dv = (1/self.dt)*(self.v - self.v0)
        du = (1/self.dt)*(self.u - self.u0)

        # write equations for fluid
        D  = sym(grad(self.v))
        D0 = sym(grad(self.v0))
        a_fluid =  inner(2*self.mu_f*D, grad(v_)) + inner(self.rho_f*grad(self.v)*self.v, v_)
        a_fluid0 = inner(2*self.mu_f*D0, grad(v_)) + inner(self.rho_f*grad(self.v0)*self.v0, v_)
       
        b_fluid  = inner(self.p, div(v_)) + inner(p_, div(self.v))
        #b_fluid0 = inner(self.p, div(v_)) + inner(p_, div(self.v))
        b_fluid0 = inner(self.p0, div(v_)) + inner(p_, div(self.v0))
        
        # TODO: clever approximation of time derivatives
        self.F_fluid = self.rho_f*inner(dv, v_)*dx \
            + Constant(self.theta)*(a_fluid*dx  + b_fluid*dx + inner(grad(self.u ), grad(u_))*dx)
        self.F_fluid0 =  Constant(1.0 - self.theta)*(a_fluid0*dx + b_fluid0*dx \
                  )#+ inner(grad(self.u0), grad(u_))*dx)

        self.F_fluid_cutcell = self.rho_f*inner(dv, v_) \
            + Constant(self.theta)*(a_fluid  + b_fluid + inner(grad(self.u ), grad(u_)))
        #self.F_fluid_cutcell = _create_dolfin_form(self.F_fluid_cutcell)
            
        self.F_fluid_cutcell0 = Constant(1.0 - self.theta)*(a_fluid0 + b_fluid0
                + inner(grad(self.u0), grad(u_)))
        #self.F_fluid_cutcell0 = _create_dolfin_form(self.F_fluid_cutcell0)

        dF_fluid = derivative(self.F_fluid, self.w, Dw)

        self.dF_fluid_cutcell = (self.rho_f/self.dt)*inner(Dv, v_) \
                + Constant(self.theta)*(inner(2*self.mu_f*sym(grad(Dv)), grad(v_)) \
                    + inner(self.rho_f*grad(Dv)*self.v, v_) \
                    + inner(self.rho_f*grad(self.v)*Dv, v_) \
                    + Dp*div(v_) + p_*div(Dv) \
                    + inner(grad(Du), grad(u_)) ) #\
        #        + (1.0 - self.theta)*( \		# implicit incompressibility+pressure
        #              Dp*div(v_)*dx + p_*div(Dv)*dx )

        #self.dF_fluid_cutcell = _create_dolfin_form(self.dF_fluid_cutcell)
    
        # write equations for solid
        E  = 0.5*(inv(self.FF).T *inv(self.FF ) - I)
        E0 = 0.5*(inv(self.FF0).T*inv(self.FF0) - I)
        # Cauchy stress 
        T_s  = JJ*inv(self.FF  )*(2*self.mu_s*E  + self.lambda_s*tr(E )*I)*inv(self.FF).T
        T_s0 = JJ0*inv(self.FF0)*(2*self.mu_s*E0 + self.lambda_s*tr(E0)*I)*inv(self.FF0).T

        a_solid  = inner((self.rho_s*JJ )*grad(self.v )*self.v,  v_) \
                + inner(grad(self.p), grad(p_))
        a_solid0 = inner((self.rho_s*JJ0)*grad(self.v0)*self.v0, v_)

        b_solid  = inner(grad(self.u )*self.v  - self.v , u_)
        b_solid0 = inner(grad(self.u0)*self.v0 - self.v0, u_)

        # FIXME proper discretization of CN-scheme:
        #self.F_solid = (self.theta*JJ+(1.0 - self.theta)*JJ0)self.rho_s*inner(dv, v_)*dx \

        self.F_solid = (self.rho_s*JJ)*inner(dv, v_)*dx + inner(du, u_)*dx \
            + Constant(self.theta)*(inner(T_s , grad(v_))*dx + a_solid*dx + b_solid*dx)
        self.F_solid0 =  Constant(1.0 - self.theta)*(inner(T_s0, grad(v_))*dx + a_solid0*dx \
                + b_solid0*dx)

        self.F_solid_cutcell= (self.rho_s*JJ)*inner(dv, v_) + inner(du, u_) \
            + Constant(self.theta)*(inner(T_s , grad(v_)) + a_solid + b_solid)
        #self.F_solid_cutcell = _create_dolfin_form(F_solid_cutcell)

        self.F_solid_cutcell0 = Constant(1.0 - self.theta)*(inner(T_s0, grad(v_)) \
                + a_solid0 + b_solid0)
        #self.F_solid_cutcell0 = _create_dolfin_form(F_solid_cutcell0)

        dF_solid = derivative(self.F_solid, self.w, Dw)

        #dE = sym(grad(Du)) + sym(grad(self.u)) + sym(grad(self.u).T*grad(Du)) + 0.5*I
        #dE = sym(grad(Du)) + sym(grad(self.u).T*grad(Du))
        dE = 0.5*inv(self.FF).T*(grad(Du).T*inv(self.FF).T + inv(self.FF)*grad(Du))*inv(self.FF)
        dF_inv = inv(self.FF)*grad(Du)*inv(self.FF)
        dJ = - JJ*tr(inv(self.FF)*grad(Du))
        dT = dJ*inv(self.FF)*(2*self.mu_s*E + self.lambda_s*tr(E)*I)*inv(self.FF).T \
                + JJ*dF_inv*(2*self.mu_s*E + self.lambda_s*tr(E)*I)*inv(self.FF).T \
                + JJ*inv(self.FF)*(2*self.mu_s*dE + self.lambda_s*tr(dE)*I)*inv(self.FF).T \
                + JJ*inv(self.FF)*(2*self.mu_s*E + self.lambda_s*tr(E)*I)*dF_inv.T
        self.dF_solid_cutcell = self.rho_s*dJ*inner(dv, v_) \
            + self.rho_s*JJ*inner(Dv/self.dt, v_) \
            + inner(Du/self.dt, u_) \
            + inner(grad(Dp), grad(p_)) \
            + Constant(self.theta)*( inner(dT, grad(v_)) \
                + self.rho_s*dJ*inner(grad(self.v )*self.v,  v_) \
                + self.rho_s*JJ*inner(grad(Dv)*self.v,  v_) \
                + self.rho_s*JJ*inner(grad(self.v)*Dv,  v_) \
                + inner(grad(Du)*self.v + grad(self.u)*Dv - Dv, u_) )
        #self.dF_solid_cutcell = _create_dolfin_form(dF_solid_cutcell)


        self.problem = Problem(self.F_fluid, self.F_fluid0, 
                 self.F_fluid_cutcell, self.F_fluid_cutcell0, 
                 self.F_solid, self.F_solid0, self.F_solid_cutcell, self.F_solid_cutcell0,
                 dF_fluid, self.dF_fluid_cutcell, dF_solid, self.dF_solid_cutcell, 
                 Dv, self.bcs, self.potentially_cutcells, self.W, self.ips, self.ips0)
        #self.problem = Problem(self.F_fluid, self.F_fluid0, 
        #         self.F_fluid_cutcell, self.F_fluid_cutcell0, 
        #         self.F_fluid, self.F_fluid0, self.F_fluid_cutcell, self.F_fluid_cutcell0,
        #         dF_fluid, self.dF_fluid_cutcell, dF_fluid, self.dF_fluid_cutcell, 
        #         self.bcs, self.potentially_cutcells, self.W, self.ips, self.ips0)


        self.solver = NewtonSolver()
        self.solver.parameters['relative_tolerance'] = 1e-6
        self.solver.parameters['absolute_tolerance'] = 1e-10
        #self.solver.parameters['maximum_iterations'] = 15
        self.solver.parameters['linear_solver']      = 'mumps'
        self.solver.parameters['error_on_nonconvergence'] = False 


        #J = derivative(F, self.w)

        # configure solver parameters
        #F = self.F_fluid + self.F_fluid0
        #self.problem = NonlinearVariationalProblem(F, self.w, bcs=self.bcs, J=dF_fluid)
        #self.solver = NonlinearVariationalSolver(self.problem) 
        #self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6 
        #self.solver.parameters['newton_solver']['linear_solver'] = 'mumps' 
 #       self.solver.parameters['newton_solver']['error_on_nonconvergence'] = False 
        #self.solver.parameters['newton_solver']['maximum_iterations'] = 15 
  
        # create files for saving 
        self.vfile = XDMFFile("%s/velo.xdmf" % result) 
        self.ufile = XDMFFile("%s/disp.xdmf" % result) 
        self.pfile = XDMFFile("%s/pres.xdmf" % result)
        self.sfile = XDMFFile("%s/stress.xdmf" % result) 
        self.ipsfile = XDMFFile("%s/ips.xdmf" % result) 
        self.vfile.parameters["flush_output"] = True 
        self.ufile.parameters["flush_output"] = True 
        self.pfile.parameters["flush_output"] = True 
        self.sfile.parameters["flush_output"] = True
        self.ipsfile.parameters["flush_output"] = True
       

    def solve(self, t, dt): 
        #info('solve')
        info('{}'.format(len(self.problem.cutcells)))
        self.t = t 
        self.dt = dt 
        self.v_in.t = t
        self.solver.solve(self.problem, self.w.vector())
        #info('solved')
        #self.solver.solve()
        self.w0.assign(self.w)
        self.ips.u0 = (project(self.u, self.U)).sub(0)
        self.ips.u1 = (project(self.u, self.U)).sub(1)
        self.ips0.u0 = (project(self.u0, self.U)).sub(0)
        self.ips0.u1 = (project(self.u0, self.U)).sub(1)
        self.ips((0.0, 0.0))
        self.ips0((0.0, 0.0)) 

        #self.problem.cutcells.clear()
        #self.problem.cutcells.update(self.problem.newcutcells)
        #self.problem.newcutcells.clear()

  #      status_solved = self.solver.solve()               # err on nonconvergence disabled         
  #      solved = status_solved[1]                         # use this control instead 
  #      if solved:
  #          (self.u, self.v, self.p) = self.w.split(True) 
  #          self.w0.assign(self.w) 
  #      return solved 

    def save(self, t):
        (v, u, p) = self.w.split() 

        ips_fun = project(self.ips, self.P)

        v.rename("v", "velocity") 
        u.rename("u", "displacement") 
        p.rename("p", "pressure") 
        self.vfile.write(v, t) 
        self.ufile.write(u, t)
        self.pfile.write(p, t) 
        self.ipsfile.write(ips_fun, t) 
        #s.rename("s", "stress") 
        #self.sfile.write(s, t)


#mesh_name = 'bench2D_L0.h5'
mesh_name = 'mesh_Euler_L2.h5'
relative_path_to_mesh = '../meshes/'+mesh_name


result = "results"		# name of folder containing results
# load mesh with boundary and domain markers
sys.path.append('../meshes/')
import marker

#(mesh, bndry, domains, interface, A, B)  \
#          = marker.give_marked_mesh(mesh_coarseness = 40, refinement = False, ALE = False)
(mesh, bndry, domains, interface, A, B) = marker.give_gmsh_mesh(relative_path_to_mesh)

# domain (used while building mesh) - needed for inflow condition(gW) and Initial Point Set function (rest)
gW   = 0.41			# width of domain
gLs1 = 0.2			# x-coordinate of centre of rigid circle
gLs2 = 0.6 + DOLFIN_EPS		# x-coordinate of end of elastic beam (in initial state)
gWs1 = 0.19 - DOLFIN_EPS	# y-coordinate of beginning of elastic beam
gWs2 = 0.21 + DOLFIN_EPS	# y-coordinate of end of elastic beam

# boundary marks' names (already setted to the mesh) - needed for boundary conditions
_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

# devide domain into 2 regions, one where is surely part of the fluid domain and second, where could elastic solid occur
regions = MeshFunction('size_t', mesh, 2, mesh.domains())
regions.set_all(0)
h = 2*mesh.hmax()	# to let some space after the end of elastic beam
for f in cells(mesh):
    mp = f.midpoint()
    if mp[0] > 0.2 and mp[0] < gLs2 + h:
        regions[f] = 1

dx  = dx(domain=mesh, subdomain_data = regions)
ds  = ds(domain=mesh, subdomain_data = bndry)

# time stepping
dt    = 0.001
t_end = 10.0
theta = 1.0
v_max = 1.5

# set material constants for elastic solids
# polybutadiene (from paper Hron&Turek)
#rho_s = Constant(910)
#nu_s  = Constant(0.5)
#E_s   = Constant(1.6e06)

# set material constants for elastic solids
# polybutadiene (from paper Hron&Turek)
#rho_s = Constant(910)
#nu_s  = Constant(0.5)
#E_s   = Constant(1.6e06)
#mu_s  = Constant(5.3e05)
# polypropylene (from paper Hron&Turek)
#rho_s = Constant(1.1e03)
#nu_s  = Constant(0.42)
#E_s   = Constant(9e08)
#mu_s  = Constant(3.17e08)

rho_s = Constant(10000)
nu_s  = Constant(0.4)
mu_s  = Constant(5e05)
E_s   = Constant(1.4e04)
lambda_s = (mu_s*E_s)/((1+mu_s)*(1-2*mu_s))

# set material constants for fluids
# glyceryne (from paper Hron&Turek)
#rho_f = Constant(1.26e03)
#nu_f  = Constant(1.13e-03)
#mu_f  = Constant(1.42)

rho_f = Constant(1.0e03)
nu_f  = Constant(1.0e-03)
mu_f  = nu_f*rho_f


################# export domains and bndry to xdmf for visualization in Paraview
# facet markers can be written directly
with XDMFFile("%s/mesh_bndry.xdmf" % result) as f:
    f.write(bndry)

#with XDMFFile("%s/mesh_interface.xdmf" % result) as f:
#    f.write(interface)

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


# check Point arithmetic
A = Point(0.1, 0.1)
B = Point(0.5, 0.2)

flow = Flow(mesh, bndry, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result)

#flow.get_integrals()

t = 0.0
while  t < t_end:
    if my_rank == 0:
        info("t = %.3f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    #end()
    flow.save(t)

    t += float(dt)

#flow.data.close()
#sys.path.append('../')
#import plotter
#result = "results"		# name of folder containing results
#plotter.plot_all_lag(result+'/data.csv', result+'/mean_press.png', result+'/pressure_jump.png', result+'/A_position.png', \
#         result+'/pressure_difference.png', result+'/drag.png', result+'/lift.png')
