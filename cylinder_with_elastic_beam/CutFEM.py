#################################################################################################
###
###     INFO:
###     - background mesh + mesh for elastic deformation (with some space around - to have
###         finer mesh here), overlapping mesh is moving over the background mesh. 
###         For this ALE move is used
###         
###     - elasticity in Lagrangian coordinates 
###     - fluid on 'elastic mesh' ALE 
###     - fluid on background mesh Eulerian
###         see: https://arxiv.org/abs/1808.00343
###
###     PROBLEMS:
###     - haven't found way to use NonlinearVariationalProblem for multimesh
###
###     - using NonlinearProblem instead, multimesh_assemble gives error
###
###     - missing interface coupling and stabilization terms
###             (I will treat it after overcoming problems with NonlinearProblem)
###
###     - disable effecting deformation by ALE movement
###
###     - find a nice way to move just the overlapping mesh
###
###     - in case of contact, the 'elastic mesh' gets out of the channel
###         * elastic mesh only for solid?
###
#################################################################################################


from dolfin import *
import mshr
import numpy as np
import csv
import sys
import os


# Use UFLACS to speed-up assembly and limit quadrature degree
#parameters['form_compiler']['representation'] = 'uflacs'   # uflacs not supported for CutFEM
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4

class Problem(NonlinearProblem):
    def __init__(self, a, L, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs
    def F(self, b, x):
        assemble_multimesh(self.L, tensor = b)		# gives "Vector cannot be initialised more than once." 
        [bc.apply(b, x) for bc in self.bcs]
    def J(self, A, x):
        assemble_multimesh(self.a, tensor = A)  	# gives "Vector cannot be initialised more than once." 
        [bc.apply(A) for bc in self.bcs]

class Flow(object):
    def __init__(self, multimesh, inflow_bndry, outflow_bndry, walls, cylinder, dt, theta, 
            v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result, *args, **kwargs):

        result = result

        self.multimesh  = multimesh
        self.dt    = dt
        self.t     = 0.0
        self.v_max = v_max

        self.mu_f     = mu_f
        self.rho_f    = rho_f
        self.lambda_s = lambda_s
        self.mu_s     = mu_s
        self.rho_f    = rho_f

        # Define finite elements
        eV1 = VectorElement("CG", self.multimesh.ufl_cell(), 1) # mesh movement space
        eV2 = VectorElement("CG", self.multimesh.ufl_cell(), 2)	# velocity space
        eU  = VectorElement("CG", self.multimesh.ufl_cell(), 2)	# displacement space
        eP  = FiniteElement("CG", self.multimesh.ufl_cell(), 1)	# pressure space
        eR  = FiniteElement("R",  self.multimesh.ufl_cell(), 0)	# Lagrange multipliers 
        eW  = MixedElement([eV2, eU, eP, eR, eR])		# function space with multipliers
        W   = MultiMeshFunctionSpace(self.multimesh, eW)
        self.W = W
        
        MV = MultiMeshFunctionSpace(self.multimesh, eV1)
        self.MV = MV

        # Create subspaces for boundary conditions
        self.V = MultiMeshSubSpace(self.W, 0)
        self.U = MultiMeshSubSpace(self.W, 1)
        self.P = MultiMeshSubSpace(self.W, 2)

        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])): \
            v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),\
                  degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        bc_u_in     = MultiMeshDirichletBC(self.U, Constant((0.0, 0.0)), inflow_bndry)
        bc_u_walls  = MultiMeshDirichletBC(self.U, Constant((0.0, 0.0)), walls)
        bc_u_out    = MultiMeshDirichletBC(self.U, Constant((0.0, 0.0)), outflow_bndry)
        bc_u_circle = MultiMeshDirichletBC(self.U, Constant((0.0, 0.0)), cylinder)
        bc_v_in     = MultiMeshDirichletBC(self.V, self.v_in,            inflow_bndry)
        bc_v_walls  = MultiMeshDirichletBC(self.V, Constant((0.0, 0.0)), walls)
        bc_v_circle = MultiMeshDirichletBC(self.V, Constant((0.0, 0.0)), cylinder)

        self.bcs = [bc_v_in, bc_v_walls, bc_v_circle, bc_u_in, bc_u_walls, bc_u_circle, bc_u_circle]

        self.n = FacetNormal(self.multimesh)
        #I = Identity(self.W.multimesh().geometry().dim())
        I = Identity(2)

        # Define test functions
        (v_, u_, p_, c_, d_) = TestFunctions(self.W)

        # current unknown time step
        self.w = MultiMeshFunction(self.W)
        (self.v, self.u, self.p, self.c, self.d) = split(self.w)
        self.mesh_u = MultiMeshFunction(self.MV)

        # previous known time step
        self.w0 = MultiMeshFunction(self.W)
        (self.v0, self.u0, self.p0, self.c0, self.d0) = split(self.w0)


        # define deformation gradient, Jacobian
        self.FF  = I + grad(self.u)
        self.FF0 = I + grad(self.u0)
        JJ  = det(self.FF)
        JJ0 = det(self.FF0)

        # approximate time derivative of dissplacement
        du = (1.0/self.dt)*(self.u - self.u0)

        # compute 1st Piola-Kirchhoff tensor for fluid
        self.S_f  = JJ *( -self.p*I  + 2*self.mu_f*sym(grad(self.v )) )*inv(self.FF).T
        self.S_f0 = JJ0*( -self.p0*I + 2*self.mu_f*sym(grad(self.v0)) )*inv(self.FF0).T

        # compute 1st Piola-Kirchhoff tensor for solid (St. Vennant - Kirchhoff model)
        B_s  = self.FF.T *self.FF
        B_s0 = self.FF0.T*self.FF0
        S_s  = self.FF *(0.5*self.lambda_s*tr(B_s  - I)*I + self.mu_s*(B_s  - I))
        S_s0 = self.FF0*(0.5*self.lambda_s*tr(B_s0 - I)*I + self.mu_s*(B_s0 - I))

        # Cauchy stress tensor
        self.T  = -self.p *I  + 2*sym(grad(self.v ))
        self.T0 = -self.p0*I  + 2*sym(grad(self.v0))

        # write equations for fluid on overlapping mesh
        a_fluid_ALE  = inner(self.S_f , grad(v_))*dX \
                + inner(JJ *self.rho_f*grad(self.v )*inv(self.FF ).T*(self.v  - du), v_)*dX
        a_fluid0_ALE = inner(self.S_f0, grad(v_))*dX \
                + inner(JJ0*self.rho_f*grad(self.v0)*inv(self.FF0).T*(self.v0 - du), v_)*dX

        b_fluid_ALE  = inner(grad(self.u ), grad(u_))*dX \
                + inner(div(JJ *inv(self.FF )*self.v),  p_)*dX
        b_fluid0_ALE = inner(grad(self.u0), grad(u_))*dX \
                + inner(div(JJ0*inv(self.FF0)*self.v0), p_)*dX

        F_fluid_ALE  = theta*(a_fluid_ALE  + b_fluid_ALE) \
                + (1.0 - theta)*(a_fluid0_ALE + b_fluid0_ALE)

        # write equation for solid
        F_solid  = theta*inner(S_s , grad(v_))*dX + (1.0 - theta)*inner(S_s0, grad(v_))*dX

        # write equations for fluid on background mesh
        a_fluid_BG  = inner(self.T , grad(v_))*dX + inner(grad(self.v )*self.v , v_)*dX
        a_fluid0_BG = inner(self.T0, grad(v_))*dX + inner(grad(self.v0)*self.v0, v_)*dX

        b_fluid_BG  = inner(div(self.v),  p_)*dX    # no artificial deformation here
        b_fluid0_BG = inner(div(self.v0), p_)*dX    # no artificial deformation here

        F_fluid_BG  = 0.5*(a_fluid_BG  + b_fluid_BG) + 0.5*(a_fluid0_BG + b_fluid0_BG)

        # discretization of temporal derivative
        F_temporal = JJ*self.rho_f*(1.0/self.dt)*inner(self.v - self.v0, v_)*dX \
                + (1.0/self.dt)*rho_s*inner(self.v - self.v0, v_)*dX \
                + inner(du, u_)*dX - ( theta*inner(self.v, u_)*dX \
                + (1.0 - theta)*inner(self.v0, u_)*dX )

        # impose zero mean value of pressure on outflow and continuity over F-S interface
        F_press = theta*(self.p*c_*ds(_OUTFLOW) + jump(self.p*d_)*dS \
                + p_*self.c*ds(_OUTFLOW) + jump(p_*self.d)*dS) \
                + (1.0 - theta)*(self.p0*c_*ds(_OUTFLOW) + jump(self.p0*d_)*dS \
                + p_*self.c0*ds(_OUTFLOW) + jump(p_*self.d0)*dS)

        # continuity (and stabilization) over F-F interface
        # (sketchy version)
        def tensor_jump(v, n):
            return outer(v('+'), n('+')) + outer(v('-'), n('-'))
        alpha = 4.0
        
        F_interface = inner(avg(self.T), tensor_jump(v_, self.n))*dI \
                    + alpha*inner(jump(self.v), jump(v_))*dI \
                    + alpha*jump(self.p * p_)*dS


        # write final equation
        F = F_temporal + F_fluid_ALE + F_fluid_BG + F_solid + F_press + F_interface

        J = derivative(F, self.w)   # works with FEniCS 2017 but not with FEniCS 2018

        #(self.v, self.u, self.p, self.c, self.d) = self.w.split(True)
        #(self.v0, self.u0, self.p0, self.c0, self.d0) = self.w0.split(True)

        self.problem = Problem(self.J, self.F, self.bcs)
        self.solver = NewtonSolver()
        self.solver.parameters['relative_tolerance'] = 1e-6
        self.solver.parameters['maximum_iterations'] = 15
        self.solver.parameters['linear_solver']      = 'mumps'

        # configure solver parameters
        #self.problem = NonlinearVariationalProblem(F, self.w, bcs=self.bcs, J=J)
        #self.solver = NonlinearVariationalSolver(self.problem)
        #self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
        #self.solver.parameters['newton_solver']['maximum_iterations'] = 15
        #self.solver.parameters['newton_solver']['linear_solver']      = 'mumps'
        PETScOptions.set('mat_mumps_icntl_24', 1)			# detects null pivots
        PETScOptions.set('mat_mumps_cntl_1', 0.01)			# set treshold for partial treshold pivoting, 0.01 is default value

        # create files for saving
        if not os.path.exists(result):
            os.makedirs(result)        
        self.vfile = XDMFFile("%s/velocity.xdmf" % result)
        self.ufile = XDMFFile("%s/displacement.xdmf" % result)
        self.pfile = XDMFFile("%s/pressure.xdmf" % result)
        self.sfile = XDMFFile("%s/stress.xdmf" % result)
        self.cfile = XDMFFile("%s/c_mult.xdmf" % result)
        self.dfile = XDMFFile("%s/d_mult.xdmf" % result)
        self.vfile.parameters["flush_output"] = True
        self.ufile.parameters["flush_output"] = True
        self.pfile.parameters["flush_output"] = True
        self.cfile.parameters["flush_output"] = True
        self.dfile.parameters["flush_output"] = True
        self.sfile.parameters["flush_output"] = True
        self.data = open(result+'/data.csv', 'w')
        self.writer = csv.writer(self.data, delimiter=';', lineterminator='\n')
        self.writer.writerow(['time', 'mean pressure on outflow', 'pressure_jump', \
                'x-coordinate of end of beam', 'y-coordinate of end of beam', 'pressure difference',\
                'drag1', 'drag2', 'lift_1', 'lift2'])



result = "results_CutFEM"		# name of folder containing results
# load mesh with boundary and domain markers
import marker
(multimesh, inflow_bndry, outflow_bndry, walls, cylinder, 
        ALE_domains, Eulerian_fluid, FS_interface, A, B) = \
        marker.give_marked_multimesh(background_coarseness = 50, 
                                     elasticity_coarseness = 40)

# domain (used while building mesh) - needed for inflow condition
gW = 0.41

# boundary marks' names (already setted to the mesh) - needed for boundary conditions
_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

# measures for integrating over single mesh
# (might come to play when trying to write equations properly)
dx_Euler = dx(domain = multimesh.part(0), subdomain_data = Eulerian_fluid)
dx_ALE   = dx(domain = multimesh.part(1), subdomain_data = ALE_domains)


# time stepping
dt    = 0.01
t_end = 10.0
theta = 0.5
v_max = 1.5

rho_s = Constant(10000)
nu_s  = Constant(0.4)
mu_s  = Constant(5e05)
E_s   = Constant(1.4e04)
lambda_s = (mu_s*E_s)/((1+mu_s)*(1-2*mu_s))

rho_f = Constant(1.0e03)
nu_f  = Constant(1.0e-03)
mu_f  = nu_f*rho_f


flow = Flow(multimesh, inflow_bndry, outflow_bndry, walls, cylinder, dt, 
        theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result)

t = 0.0

while  t < t_end:
    info("t = %.3f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    #end()
    flow.save(t)

    t += float(dt)
