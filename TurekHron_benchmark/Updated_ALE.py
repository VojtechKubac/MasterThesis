"""
This code was implemented by Vojtech Kubac as a part of his Master Thesis that will be defended
on February 2020. At the Faculty of Mathematics and Physics of Charles University in Prague.
"""

"""
This code solves FSI2 and FSI3 Benchmarks from
    "S. Turek and J. Hron, “Proposal for numerical benchmarking of fluid–
     structure interaction between an elastic object and laminar incompress-
     ible flow,” in Fluid-Structure Interaction - Modelling, Simulation, Opti-
     mization, ser. Lecture Notes in Computational Science and Engineering,"

other FSI simulations can be run by loading corresponding mesh and a straightforward 
modifications of boundary conditions.

The equations are written in Updated ALE formulation, where for the mesh movement pseudoelasticity
extension of the solid displacement was used.

Chosen Finite Elements are linear discontinuous space for pressure and quadratic continuous space
enriched with quadratic bubble is used for displacement and velocity.

Time discretization uses PETSc time stepping methods, preferably BDF, BE and CN.
"""

import mshr
import numpy as np
import csv
import sys
import os.path
import time
from mpi4py.MPI import COMM_WORLD
from optparse import OptionParser

from dolfin import *
from dolfin import __version__


# mpi4py communicator
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


class VelocityProblem(NonlinearProblem):
    def __init__(self, F_fluid, F_solid, dF_fluid, dF_solid, bcs_press, bcs):
        NonlinearProblem.__init__(self)
        self.F_fluid = F_fluid
        self.F_solid = F_solid
        self.dF_fluid = dF_fluid
        self.dF_solid = dF_solid
        self.bcs_press = bcs_press
        self.bcs = bcs
        self.assembler = SystemAssembler(dF_solid+dF_fluid, F_solid+F_fluid, bcs+bcs_press) 
        self.A1 = PETScMatrix()
        self.A2 = PETScMatrix()

    def F(self, b, x):
        #info("Assemble b.")
        self.assembler.init_global_tensor(b, Form(self.F_solid+self.F_fluid))
        b.apply('add')
        
        b1=Vector()
        b2=Vector()
        #info("Assemble b1.")
        assemble(self.F_solid, tensor = b1)
        [bc.apply(b1) for bc in self.bcs_press]

        #info("Assemble b2.")
        assemble(self.F_fluid, tensor = b2)
                                                        
        #print(b,b1,b2)
        b.axpy(1,b1)
        b.axpy(1,b2)
        #info("Apply bc to b.")
        [bc.apply(b, x) for bc in self.bcs]
        
    def J(self, A, x):
        #info("Assemble A.")
        self.assembler.init_global_tensor(A, Form(self.dF_solid+self.dF_fluid))
        A.apply('insert')

        #info("Assemble A1.")
        assemble(self.dF_solid, tensor = self.A1, keep_diagonal=True)
        #info("Apply bc to A1.")
        [bc.zero(self.A1) for bc in self.bcs_press]
        #info("Assemble A2.")
        assemble(self.dF_fluid, tensor = self.A2, keep_diagonal=True)

        A.axpy(1, self.A1, False)
        A.axpy(1, self.A2, False)
        #info("Apply bc to A.")
        [bc.apply(A) for bc in self.bcs]



class DisplacementProblem(NonlinearProblem):
    """
    frommknown velocities computes displacement
    """
    def __init__(self, F_solid, F_mesh, bcs_mesh, bcs):
        self.a_mesh = lhs(F_mesh)
        self.l_mesh = rhs(F_mesh)   # d part of the equations
        self.a_solid = lhs(F_solid)
        self.l_solid = rhs(F_solid) # the velocity part
        self.bcs_mesh = bcs_mesh
        self.bcs = bcs

        #info('{}'.format(type(self.l_mesh)))
        #info('{}'.format(type(self.l_solid)))
        self.A2 = assemble(self.a_mesh, keep_diagonal=True)
        [bc.zero(self.A2) for bc in self.bcs_mesh]


    def solve(self, u):

        # assemble equation for solid disaplecement (from velocity)
        A1 = assemble(self.a_solid, keep_diagonal=True)
        b1 = assemble(self.l_solid)

        #[bc.apply(A1, b1) for bc in self.bcs]
        #solve(A1, u.vector(), b1)

        # assemble mesh-moving equations from initial configuration
        #self.A2 = assemble(self.a_mesh, keep_diagonal=True)
        #[bc.zero(self.A2) for bc in self.bcs_mesh]

        b2 = assemble(self.l_mesh)
        [bc.apply(b2) for bc in self.bcs_mesh]

        #[bc.apply(A2, b2) for bc in self.bcs] 

        A = A1 + self.A2
        b = b1# + b2
        [bc.apply(A, b) for bc in self.bcs]
        solve(A, u.vector(), b, "mumps") # "superlu_dist") #

        return u


class Flow(object):
    def __init__(self, mesh, bndry, interface, dt, theta, v_max, lambda_s, mu_s, rho_s, 
                 mu_f, rho_f, result, restart, *args, **kwargs):

        self.init_time = time.time()

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

        self.result = result

        self.A = A
        self.B = B

        # bounding box tree
        self.bb = BoundingBoxTree()
        self.bb.build(self.mesh)

        # Define finite elements
        eU_mesh = VectorElement("CG", self.mesh.ufl_cell(), 1)       	# mesh movement space
        eV = VectorElement("CG", self.mesh.ufl_cell(), 2)		# velocity space
        eB = VectorElement("Bubble", self.mesh.ufl_cell(), mesh.geometry().dim()+1) # Bubble element
        eU = VectorElement("CG", self.mesh.ufl_cell(), 2)		# displacement space
        eP = FiniteElement("DG", self.mesh.ufl_cell(), 1)		# pressure space

        eW = MixedElement([eV, eB, eP])
        W  = FunctionSpace(self.mesh, eW)
        self.W = W
        self.U = FunctionSpace(self.mesh, MixedElement([eU, eB]))

        MU = FunctionSpace(self.mesh, eU_mesh)
        self.MU = MU

        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])):\
                      v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),\
                      degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        bc_u_in     = DirichletBC(self.U.sub(0), Constant((0.0, 0.0)), bndry, _INFLOW)
        bc_u_walls  = DirichletBC(self.U.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_u_out    = DirichletBC(self.U.sub(0), Constant((0.0, 0.0)), bndry, _OUTFLOW)
        bc_u_circle = DirichletBC(self.U.sub(0), Constant((0.0, 0.0)), bndry, _CIRCLE)
        bc_v_in     = DirichletBC(self.W.sub(0), self.v_in,            bndry, _INFLOW)
        bc_v_walls  = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_v_circle = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _CIRCLE)

        self.bcs_velo = [bc_v_in, bc_v_walls, bc_v_circle]
        self.bcs_disp = [bc_u_in, bc_u_walls, bc_u_circle, bc_u_out]

        bc_mesh = DirichletBC(self.U.sub(0), Constant((0.0, 0.0)), interface, _FSI)
        self.bcs_mesh = [bc_mesh]

        bc_press    = DirichletBC(self.W.sub(2), Constant(0.0),        bndry, _FSI)
        self.bcs_press = [bc_press]


        self.n = FacetNormal(self.mesh)
        self.h = Circumradius(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())


        # Define functions
        self.w  = Function(self.W)      
        self.u_fun = Function(self.U)
        self.w0 = Function(self.W)
        self.u_fun0 = Function(self.U)

        (v__, bv_, p_) = TestFunctions(self.W)
        v_ = v__ + bv_
        u__fun = TestFunction(self.U)
        (u__, bu_) = split(u__fun)
        u_ = u__ + bu_
        (ut_nodal, utb) = TrialFunctions(self.U)
        ut = ut_nodal + utb
        (v, bv, self.p) = split(self.w)
        self.v = v + bv
        (u, bu) = split(self.u_fun)
        self.u = u + bu
        (v0, bv0, self.p0) = split(self.w0)
        self.v0 = v0 + bv0

        # displacement w.r.t. initial time
        self.d_fun = Function(self.U)
        (self.d_nodal, self.bd) = split(self.d_fun)
        self.d = self.d_nodal + self.bd

        # ALE velocity
        self.v_ALE_fun = Function(self.U)
        (self.v_ALE_nodal, self.bv_ALE) = split(self.v_ALE_fun)
        self.v_ALE = self.v_ALE_nodal + self.bv_ALE

        # mesh displacement
        self.mesh_u = Function(self.MU)

        # define deformation gradient, Jacobian
        FF_d = I - grad(self.d)         # def. grad. from config in time t onto initial
        #FF_d = I
        JJ_d = det(FF_d)

        FF0 = inv(FF_d)			# = I + grad(\hat{d}), def.grad. from initial time

        theta_du = Constant(0.5)
        grad_u = self.dt*grad(theta_du*self.v + (1.0 - theta_du)*self.v0)
        FF  = (I + grad_u)*FF0

        def Grad(f, F): return grad(f)*inv(F)
        def Div(f, F): return tr( Grad(f, F) )

        # write mesh-moving equation
        self.gamma = 9.0/8.0 # 0.98 # 1.0 #
        h = CellVolume(self.mesh)**(self.gamma)
        E = Constant(1.0)

        E_mesh = E/h
        nu_mesh = Constant(0.1)

        mu_mesh = E_mesh/(2*(1.0+nu_mesh))
        lambda_mesh = (nu_mesh*E_mesh)/((1+nu_mesh)*(1-2*nu_mesh))

        #FF_mesh = (I + grad(ut))*FF0 - I
        FF_mesh = grad(ut) + grad(self.d)

        F_mesh = inner(mu_mesh*2*sym(FF_mesh), grad(u_))*dx(0) \
                          + lambda_mesh*inner(tr(FF_mesh), div(u_))*dx(0)

        # approximate time derivative of dissplacement
        #self.du0 = (1.0/self.dt)*self.u0        # will be updated in the end of every time step
        #theta_u = Constant(1.0)			# discretization of mesh velocity
        self.du = (1.0/self.dt)*self.u          # u0 is allways zero
        #self.v_ALE = theta_u*self.du + (1.0 - theta_u)*self.du0		# theta scheme for mesh velocity
        dv = (1.0/self.dt)*(self.v - self.v0)

        # compute the Cauchy stress tensor for fluid
        self.T_f  = -self.p*I  + 2*self.mu_f*sym(grad(self.v))
        self.T_f0 = -self.p0*I + 2*self.mu_f*sym(grad(self.v0))

        # Compute 1st Piola-Kirhhoff tensor for fluid 
        #       - for computing surface integrals for forces in postprocessing 
        self.S_f = self.T_f

        # write equations for fluid
        a_fluid  = inner(self.T_f , grad(v_))*dx(0) \
               + inner(self.rho_f*grad(self.v)*(self.v  - self.v_ALE), v_)*dx(0)
        a_fluid0 = inner(self.T_f0, grad(v_))*dx(0) \
               + inner(self.rho_f*grad(self.v0)*(self.v0 - self.v_ALE), v_)*dx(0)

        b_fluid  = inner(div(self.v), p_)*dx(0)
        b_fluid0 = inner(div(self.v), p_)*dx(0)

        self.F_fluid  = self.rho_f*inner(dv, v_)*dx(0) \
                   + self.theta*(a_fluid  + b_fluid) + (1.0 - self.theta)*(a_fluid0 + b_fluid0)
                   #+ F_mesh

        # compute 1st Piola-Kirchhoff tensor for solid (St. Vennant - Kirchhoff model)
        E_s  = 0.5*(FF.T*FF - I)
        E_s0 = 0.5*(FF0.T*FF0 - I)
        S_s  = JJ_d*FF*(2*self.mu_s*E_s + self.lambda_s*tr(E_s)*I)
        S_s0 = JJ_d*FF0*(2*self.mu_s*E_s0 + self.lambda_s*tr(E_s0)*I)

        # write equation for solid
        self.F_solid = self.rho_s*JJ_d*inner(dv, v_)*dx(1) \
                   + self.theta*inner(S_s , Grad(v_, FF_d))*dx(1) \
                   + (1.0 - self.theta)*inner(S_s0, Grad(v_, FF_d))*dx(1) \
                   + inner(grad(self.p), grad(p_))*dx(1)

        theta_v = Constant(0.5)
        F_disp = inner(ut/self.dt - (theta_v*self.v + (1.0 - theta_v)*self.v0), u_)*dx(1)

        #F = self.F_fluid + self.F_solid
        #dF = derivative(F, self.w)

        dF_fluid = derivative(self.F_fluid, self.w)
        dF_solid = derivative(self.F_solid, self.w)

        #self.problem = Problem(self.F_fluid, self.F_solid, dF_fluid, dF_solid, self.bcs_mesh, self.bcs)
        #self.velocityProblem = NonlinearVariationalProblem(F, self.w, bcs=self.bcs_velo, J=dF)
        #self.velocitySolver = NonlinearVariationalSolver(self.velocityProblem)
        self.velocityProblem = VelocityProblem(self.F_fluid, self.F_solid, dF_fluid, dF_solid, 
                                          self.bcs_press, self.bcs_velo)
        self.velocitySolver = NewtonSolver()

        # configure solver parameters
        self.velocitySolver.parameters['relative_tolerance'] = 1e-6
        self.velocitySolver.parameters['absolute_tolerance'] = 2e-10
        self.velocitySolver.parameters['maximum_iterations'] = 15
        self.velocitySolver.parameters['linear_solver']      = 'mumps'

        self.displacementProblem = DisplacementProblem(F_disp, F_mesh, self.bcs_mesh, self.bcs_disp)

        #self.velocitySolver.parameters['newton_solver']['relative_tolerance'] = 1e-6
        #self.velocitySolver.parameters['newton_solver']['maximum_iterations'] = 15
        #self.velocitySolver.parameters['newton_solver']['linear_solver']      = 'mumps'
        #self.velocitySolver.parameters['newton_solver']['error_on_nonconvergence'] = True       

        # configure solver parameters
        #self.velocitySolver.parameters['nonlinear_solver'] = 'snes'
        #self.velocitySolver.parameters['snes_solver']['linear_solver'] = 'mumps'
        #self.velocitySolver.parameters['snes_solver']['absolute_tolerance'] = 1E-8
        #self.velocitySolver.parameters['snes_solver']['relative_tolerance'] = 1e-8
        #self.velocitySolver.parameters['snes_solver']['maximum_iterations'] = 10
        #self.velocitySolver.parameters['snes_solver']['report'] = True
        #self.velocitySolver.parameters['snes_solver']['error_on_nonconvergence'] = False
        #self.velocitySolver.parameters['snes_solver']['method'] = 'newtonls'
        #self.velocitySolver.parameters['snes_solver']['line_search'] = 'bt'

        # create files for saving
        if my_rank == 0:
            if not os.path.exists(result):
                os.makedirs(result)

        mode = 'a' if restart else 'w'
        with open(result+'/data.csv', mode) as data_file:
            writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
            writer.writerow(['time', 'mean pressure on outflow', 'pressure_jump', 
                              'x-coordinate of end of beam', 'y-coordinate of end of beam',
                              'pressure difference', 
                              'drag_circle', 'drag_fluid', 'drag_solid', 'drag_fullfluid',
                              'lift_circle', 'lift_fluid', 'lift_solid', 'lift_fullfluid',
                              'runtime_before_nonlinear_solve',
                              'runtime_after_nonlinear_solve', 'runtime_after_linear_solve'])

        # find dofs for evaluating displacement in A
        self.dof_pA = -1
        self.dof_pB = -1
        self.dof_Ax = -1
        self.dof_Ay = -1

        dofmap = self.W.dofmap()

        # collect cells containing A (resp. B)
        cells_A = self.bb.compute_collisions(A)
        cells_B = self.bb.compute_collisions(B)


        # finf dofs Ax, Ay (from displacement)
        from statistics import collections
        if len(cells_A):
            # Among dofs of those cells find one whose coordinates are closest to A

            # collect all dofs
            dofmap_ux = self.U.sub(0).sub(0).dofmap()
            dofs_ux = dofmap_ux.cell_dofs(cells_A[0])
            dofmap_uy = self.U.sub(0).sub(1).dofmap()
            dofs_uy = dofmap_uy.cell_dofs(cells_A[0])

            for i in range(1, len(cells_A)):
                dofs_ux = np.concatenate((dofs_ux, dofmap_ux.cell_dofs(cells_A[i])))
                dofs_uy = np.concatenate((dofs_uy, dofmap_uy.cell_dofs(cells_A[i])))

            # order the lists by frequency of items and extract first element 
            counts = collections.Counter(dofs_ux)
            dof_ = sorted(dofs_ux, key=lambda x: -counts[x])[0]		# most frequented item of the list
            if dof_ < len(self.d_fun.vector()):	# check if the dof belongs to current process
                self.dof_Ax = dof_
            counts = collections.Counter(dofs_uy)
            dof_ = sorted(dofs_uy, key=lambda x: -counts[x])[0]		# most frequented item of the list
            if dof_ < len(self.d_fun.vector()):	# check if the dof belongs to current process:
                self.dof_Ay = dof_


        # find dofs for pressure evaluation
        #     discrete space - need to solve variational problem
        f_px = Expression('x[0]', degree=1)
        f_py = Expression('x[1]', degree=1)

        (vv, _, pp) = TrialFunctions(self.W)
        F_px = pp*p_*dx - f_px*p_*dx
        F_py = pp*p_*dx - f_py*p_*dx

        solve(lhs(F_px) == rhs(F_px), self.w)
        if len(cells_A):
            dofmap_pA = self.W.sub(2).dofmap()
            dofs_pA = dofmap_pA.cell_dofs(cells_A[0])

            for dof in dofs_pA:
                info('{}'.format(self.w.vector()[dof]))
                if  not near(self.w.vector()[dof], 0.6):
                    dofs_pA = np.delete(dofs_pA, np.argwhere((dofs_pA == dof)))

        if len(cells_B):
            dofmap_pB = self.W.sub(2).dofmap()
            dofs_pB = dofmap_pB.cell_dofs(cells_B[0])

            for dof in dofs_pB:
                if  not near(self.w.vector()[dof], 0.15):
                    dofs_pB = np.delete(dofs_pB, np.argwhere((dofs_pB == dof)))

        solve(lhs(F_py) == rhs(F_py), self.w)

        info('\n')
        if len(cells_A):
            for dof in dofs_pA:
                info('{}'.format(self.w.vector()[dof]))
                if  not near(self.w.vector()[dof], 0.2):
                    dofs_pA = np.delete(dofs_pA, np.argwhere((dofs_pA == dof)))

            if len(dofs_pA) == 0:
                raise RuntimeError("Failed to find dof for pA.")
            else:
                self.dof_pA = dofs_pA[0]

        if len(cells_B):
            for dof in dofs_pB:
                if  not near(self.w.vector()[dof], 0.2):
                    dofs_pB = np.delete(dofs_pB, np.argwhere((dofs_pB == dof)))

            if len(dofs_pB) == 0:
                raise RuntimeError("Failed to find dof for pB.")
            else:
                self.dof_pB = dofs_pB[0]

        N = len(self.w.vector())
        self.w.vector()[:] = np.zeros(N)
        self.Ax = 0.0
        self.Ay = 0.0

        self.w_prev = Function(self.W)
        (v_prev_nod, v_prev_b, self.p_prev) = split(self.w_prev)
        self.v_prev = v_prev_nod + v_prev_b

        self.file_hdf=HDF5File(comm,self.result+"/solution.h5", mode)
        self.file={}
        for i in ['v', 'u', 'p']:
            self.file[i] = XDMFFile(self.result+"/{}.xdmf".format(i))
            self.file[i].parameters["flush_output"] = True
            self.file[i].parameters["functions_share_mesh"] = True

    def read_solution(self) :
        name=self.result+"/solution.h5"
        with HDF5File(comm, name, 'r') as hdf:
            nstep = hdf.attributes("/w")['count'] -1
            idw="/w/vector_{0:d}".format(nstep)
            self.t=hdf.attributes(idw)['timestamp']
            hdf.read(self.w, idw)
            self.w0.assign(self.w)
        info("RESTART: at t={0}  i={1}".format(self.t, nstep))
        
        return self.t, nstep
        

    def solve(self, t, dt):
        self.t = t
        self.v_in.t = t
        self.dt = Constant(dt)

        self.time_t0 = time.time() - self.init_time
        its, solved = self.velocitySolver.solve(self.velocityProblem, self.w.vector())
        #status_solved = self.velocitySolver.solve()  # err on nonconvergence disabled
        #solved = status_solved[1]                          # use this control instead
        self.time_t1 = time.time() - self.init_time

        self.u_fun = self.displacementProblem.solve(self.u_fun)
        self.time_t2 = time.time() - self.init_time

        self.d_fun.vector()[:] = self.d_fun.vector()[:] + self.u_fun.vector()[:]
        self.v_ALE_fun.assign(Constant(1.0/self.dt)*self.u_fun)
                     #FIXME: consider theta*u + (1-theta)u0
        self.w0.assign(self.w)

        (u_nodal, bu) = self.u_fun.split(True)
        self.mesh_u = interpolate(u_nodal, self.MU)

        return solved


    def save(self, t):
        (v, bv, p) = self.w.split()
        (d, bd) = self.d_fun.split()

        # save solution to a hdf file for restart
        if t < 8.0:
            self.file_hdf.write(self.w,"/w",t)
            self.file_hdf.flush()

        #r = Function(self.W)
        #assemble(self.R, tensor=r)
        #for bc in self.bcs : bc.apply(r, u=self.w)
        info("solution: |u|, |v|, |p| = {{{0:.2e}, {1:.2e}, {2:.2e}}},".format(
                                                     norm(d), norm(v), norm(p)))

        v.rename("v", "velocity")
        d.rename("u", "displacement")
        p.rename("p", "pressure")

        self.file['v'].write(v, t)
        self.file['u'].write(d, t)
        self.file['p'].write(p, t)
        v.rename("v", "velocity")
        d.rename("u", "displacement")
        p.rename("p", "pressure")

        P = assemble(self.p*ds(_OUTFLOW))/gW
        PI  = assemble(abs(jump(self.p))*dS(1))

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

#        self.d.set_allow_extrapolation(True)
#        self.p.set_allow_extrapolation(True)
#        pA_loc = self.p((self.A.x(), self.A.y()))
#        pB_loc = self.p((B.x(), B.y()))
#        Ax_loc = self.d[0]((self.A.x(), self.A.y()))
#        Ay_loc = self.d[1]((self.A.x(), self.A.y()))
#        self.p.set_allow_extrapolation(False)
#        self.d.set_allow_extrapolation(False)
#
#        pi = 0
#        if self.bb.compute_first_collision(self.A) < 4294967295:
#            pi = 1
#        else:
#            pA_loc = 0.0
#            Ax_loc = 0.0
#            Ay_loc = 0.0
#        pA = MPI.sum(comm, pA_loc) / MPI.sum(comm, pi)
#        Ax = MPI.sum(comm, Ax_loc) / MPI.sum(comm, pi)
#        Ay = MPI.sum(comm, Ay_loc) / MPI.sum(comm, pi)
#
#        self.A = Point(A.x() + Ax, A.y() + Ay)
#
#        pi = 0
#        if self.bb.compute_first_collision(B) < 4294967295:
#            pi = 1
#        else:
#            pB_loc = 0.0
#        pB = MPI.sum(comm, pB_loc) / MPI.sum(comm, pi)
# doesn't work, I use approach with dofs instead

        pi = 0
        if self.dof_Ax != -1:
            Ax_loc = self.Ax + self.d_fun.vector()[self.dof_Ax]
            pi += 1
        else:
            Ax_loc = 0.0
        self.Ax = MPI.sum(comm, Ax_loc) / MPI.sum(comm, pi)

        pi = 0
        if self.dof_Ay != -1:
            Ay_loc = self.Ay + self.d_fun.vector()[self.dof_Ay]
            pi += 1
        else:
            Ay_loc = 0.0
        self.Ay = MPI.sum(comm, Ay_loc) / MPI.sum(comm, pi)

        pi = 0
        if self.dof_pA != -1:
            pA_loc = self.w.vector()[self.dof_pA]
            pi += 1
        else:
            pA_loc = 0.0
        pA = MPI.sum(comm, pA_loc) / MPI.sum(comm, pi)

        pi = 0
        if self.dof_pB != -1:
            pB_loc = self.w.vector()[self.dof_pB]
            pi += 1
        else:
            pB_loc = 0.0
        pB = MPI.sum(comm, pB_loc) / MPI.sum(comm, pi)

        p_diff = pB - pA

        if my_rank == 0:
            with open(result+'/data.csv', 'a') as data_file:
                writer = csv.writer(data_file, delimiter=';', lineterminator='\n')
                writer.writerow([t, P, PI, self.Ax, self.Ay, p_diff, 
                              D_C, D_F, D_S, D_FF, L_C, L_F, L_S, L_FF,
                              self.time_t0, self.time_t1, self.time_t2])

        # move mesh
        ALE.move(self.mesh, self.mesh_u)
        self.bb.build(self.mesh)


    def get_integrals(self):
        self.area        = assemble(1*dx)
        area_fluid       = assemble(1*dx(0))
        area_solid       = assemble(1*dx(1))
        elastic_surface  = assemble(1*dS(1))
        solid_surface    = elastic_surface + assemble(1*dss(1))

        if my_rank == 0:
            info("area of the whole domain = {}, \narea of fluid domain = {},".format(self.area,
                                                                                   area_fluid ))
            info("area of solid domain = {}, \nelastic surface = {}, \nsolid surface = {}".format(
                                        area_solid, elastic_surface, solid_surface))
            info("Degrees of freedom = {}".format(self.W.dim()))
                 
def get_benchmark_specification(benchmark = 'FSI1'):
    if benchmark == 'FSI1':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 0.2
        T_end = 40.0
        result = "results-FSI1_part/"
    elif benchmark == 'FSI2':
        rho_s = Constant(1e04)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 1.0
        T_end = 15.0
        result = "results-FSI2_part/"
    elif benchmark == 'FSI3':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(2e06)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 2.0
        T_end = 20.0
        result = "results-FSI3_part/"
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
parser.add_option("--mesh_move", dest="mesh_move", default='pseudoelasticity')
parser.add_option("--theta", dest="theta", default='1.0')
parser.add_option("--dt", dest="dt", default='0.001')
parser.add_option("--restart", dest="restart", default=False, action="store_true")
#parser.add_option("--dt_scheme", dest="dt_scheme", default='CN')	# BE BE_CN

(options, args) = parser.parse_args()

# name of benchmark 
benchmark = options.benchmark

# name of mesh
mesh_name = options.mesh_name
relative_path_to_mesh = '../meshes/'+mesh_name+'.h5'

# approah to mesh moving in fluid region
mesh_move = options.mesh_move

# value of theta to theta scheme for temporal discretization
theta = Constant(options.theta)

# time step size
dt = options.dt

# continue with computations
restart = options.restart

# time stepping scheme
#dt_scheme = options.dt_scheme


# load mesh with boundary and domain markers
sys.path.append('../meshes/')
import marker

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

v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, t_end, result = get_benchmark_specification(benchmark)
#result = result + 'dt_' + str(dt) + '/' + dt_scheme + '/' + mesh_name[:-3] + '/' + mesh_name[-2:]
if theta.values()[0] == 1.0:
    result = result + 'dt_' + str(dt) + '/BE/' + mesh_name[:-3] + '/' + mesh_name[-2:]
elif theta.values()[0] == 0.5:
    result = result + 'dt_' + str(dt) + '/CN/' + mesh_name[:-3] + '/' + mesh_name[-2:]
else:
    raise RuntimeError('Theta is not 1.0 or 0.5, unable to create folders.')

flow = Flow(mesh, bndry, interface, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, 
               result, restart)

flow.get_integrals()

t = 0.0

if not restart :
    info("Save the hdf mesh and boundaries...")
    flow.hdf = HDF5File(mesh.mpi_comm(), result+"/mesh.h5", "w")
    flow.hdf.write(mesh, "/mesh")
    if(domains) : flow.hdf.write(domains, "/domains")
    if(bndry) : flow.hdf.write(bndry, "/bndry")
    flow.hdf.close()
    info("Done.")
else:
    t,nstep = flow.read_solution()

while  t < t_end:
    if my_rank == 0: 
        info("t = %.4f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    flow.save(t)

    t += float(dt)
