###############################################################
###
###     Get rid of influence of artificial movement of fluid domain 
###         on deformation of elastic solid.
###
###############################################################
from dolfin import *
import mshr
import numpy as np
import csv
import sys
import os


# Use UFLACS to speed-up assembly and limit quadrature degree
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4


class Flow(object):
    def __init__(self, mesh, bndry, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result, 
            *args, **kwargs):

        result = result

        self.mesh  = mesh
        self.bndry = bndry
        self.dt    = dt
        self.t     = 0.0
        self.v_max = v_max

        self.mu_f     = mu_f
        self.rho_f    = rho_f
        self.lambda_s = lambda_s
        self.mu_s     = mu_s
        self.rho_f    = rho_f


        # Define finite elements
        eV1 = VectorElement("CG", mesh.ufl_cell(), 1)           # mesh movement space
        eV2 = VectorElement("CG", mesh.ufl_cell(), 2)		# velocity space
        eU  = VectorElement("CG", mesh.ufl_cell(), 2)		# displacement space
        eP  = FiniteElement("CG", mesh.ufl_cell(), 1)		# pressure space
        eR = FiniteElement("R",  mesh.ufl_cell(), 0)		# for Lagrange multipliers for zero mean value of pressure on outflow 
                                                                # and continuity of pressure on the interface
        eW = MixedElement([eV2, eU, eP, eR, eR])			# function space with multipliers
        W   = FunctionSpace(self.mesh, eW)
        self.W = W

        MV = FunctionSpace(self.mesh, eV1)
        self.MV = MV


        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])): \
                v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),\
                  degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        bc_u_in     = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _INFLOW)
        bc_u_walls  = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_u_out    = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _OUTFLOW)
        bc_u_circle = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _CIRCLE)
        bc_v_in     = DirichletBC(self.W.sub(0), self.v_in,            bndry, _INFLOW)
        bc_v_walls  = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_v_circle = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _CIRCLE)

        self.bcs = [bc_v_in, bc_v_walls, bc_v_circle, bc_u_in, bc_u_walls, bc_u_circle, bc_u_circle]

        self.n = FacetNormal(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())

        # Define test functions
        (v_, u_, p_, c_, d_) = TestFunctions(self.W)

        # current unknown time step
        self.w = Function(self.W)
        (self.v, self.u, self.p, self.c, self.d) = split(self.w)
        self.mesh_u = Function(self.MV)

        # previous known time step
        self.w0 = Function(self.W)
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

        # write equations for fluid
        a_fluid  = inner(self.S_f , grad(v_))*dx(0) \
                + inner(JJ *self.rho_f*grad(self.v )*inv(self.FF ).T*(self.v  - du), v_)*dx(0)
        a_fluid0 = inner(self.S_f0, grad(v_))*dx(0) \
                + inner(JJ0*self.rho_f*grad(self.v0)*inv(self.FF0).T*(self.v0 - du), v_)*dx(0)

        b_fluid  = inner(grad(self.u ), grad(u_))*dx(0) + inner(div(JJ *inv(self.FF )*self.v),  p_)*dx(0)
        b_fluid0 = inner(grad(self.u0), grad(u_))*dx(0) + inner(div(JJ0*inv(self.FF0)*self.v0), p_)*dx(0)

        F_fluid  = theta*(a_fluid  + b_fluid) + (1.0 - theta)*(a_fluid0 + b_fluid0)

        # compute 1st Piola-Kirchhoff tensor for solid (St. Vennant - Kirchhoff model)
        B_s  = self.FF.T *self.FF
        B_s0 = self.FF0.T*self.FF0
        S_s  = self.FF *(0.5*self.lambda_s*tr(B_s  - I)*I + self.mu_s*(B_s  - I))
        S_s0 = self.FF0*(0.5*self.lambda_s*tr(B_s0 - I)*I + self.mu_s*(B_s0 - I))

        # write equation for solid
        F_solid  = theta*inner(S_s , grad(v_))*dx(1) + (1.0 - theta)*inner(S_s0, grad(v_))*dx(1)

        # discretization of temporal derivative
        F_temporal = JJ*self.rho_f*(1.0/self.dt)*inner(self.v - self.v0, v_)*dx(0) \
                + (1.0/self.dt)*rho_s*inner(self.v - self.v0, v_)*dx(1) \
                + inner(du, u_)*dx(1) - ( theta*inner(self.v, u_)*dx(1) \
                + (1.0 - theta)*inner(self.v0, u_)*dx(1) )

        # impose zero mean value of pressure on outflow and continuity over interface
        F_press = theta*(self.p*c_*ds(_OUTFLOW) + jump(self.p*d_)*dS(1) \
                + p_*self.c*ds(_OUTFLOW) + jump(p_*self.d)*dS(1)) \
                + (1.0 - theta)*(self.p0*c_*ds(_OUTFLOW) + jump(self.p0*d_)*dS(1) \
                + p_*self.c0*ds(_OUTFLOW) + jump(p_*self.d0)*dS(1))

        # write final equation
        F = F_temporal + F_fluid + F_solid + F_press

        J = derivative(F, self.w)

        (self.v, self.u, self.p, self.c, self.d) = self.w.split(True)
        (self.v0, self.u0, self.p0, self.c0, self.d0) = self.w0.split(True)

        # configure solver parameters
        self.problem = NonlinearVariationalProblem(F, self.w, bcs=self.bcs, J=J)
        self.solver = NonlinearVariationalSolver(self.problem)
        self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
        self.solver.parameters['newton_solver']['maximum_iterations'] = 15
        self.solver.parameters['newton_solver']['linear_solver']      = 'mumps'
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


    def solve(self, t, dt):
        self.t = t
        self.v_in.t = t
        self.dt = dt
        self.solver.solve()
        (self.v, self.u, self.p, self.c, self.d) = self.w.split()

        # mesh movement
        u = Function(self.MV)
        u0 = Function(self.MV)
        u.assign(interpolate(self.u, self.MV))
        u0.assign(interpolate(self.u0, self.MV))
        self.mesh_u.assign(u - u0)
        ALE.move(self.mesh, self.mesh_u)
        self.mesh.bounding_box_tree().build(self.mesh)

        self.w0.assign(self.w)
        (self.v0, self.u0, self.p0, self.c0, self.d0) = self.w0.split(True)

    def save(self, t):
        (v, u, p, c, d) = self.w.split()

        v.rename("v", "velocity")
        u.rename("u", "displacement")
        p.rename("p", "pressure")
        c.rename("c", "multiplier_mean")
        d.rename("d", "multiplier_jump")
        self.vfile.write(v, t)
        self.ufile.write(u, t)
        self.pfile.write(p, t)
        self.cfile.write(c, t)
        self.dfile.write(d, t)
        P = assemble(self.p*ds(_OUTFLOW))/gW
        PI  = assemble(abs(jump(self.p))*dS(1))

        def dvtdn(grad_v, n):
            return dot(grad_v*as_vector([n[1], -n[0]]), n)

        def integrate_over_interface(f):
            return assemble(f*dss(1) + f('+')*dS(1))

        def give_drag_and_lift(grad_v, n, p):
            drag =   integrate_over_interface(self.rho_f*self.mu_f*dvtdn(grad_v, n)*n[1] - p*n[0])
            lift = - integrate_over_interface(self.rho_f*self.mu_f*dvtdn(grad_v, n)*n[0] + p*n[1])
            return(drag, lift)

        # compute drag and lift
        (drag_Lagrangian, lift_Lagrangian) = give_drag_and_lift(grad(self.v), self.n, self.p)

        S0 = integrate_over_interface((self.S_f*self.n)[0])
        S1 = integrate_over_interface((self.S_f*self.n)[1])
        pA = self.p((A.x(), A.y()))
        pB = self.p((B.x(), B.y()))
        p_diff = pB - pA
        Ax = A.x() + self.u[0]((A.x(), A.y()))
        Ay = A.y() + self.u[1]((A.x(), A.y()))
        #info("{}, {}, {}, {}, {}, {}, {}, {}, {}".format\
        #       (t, P, PI, Ax, Ay, p_diff, drag_Lagrangian, S0, lift_Lagrangian, S1))
        self.writer.writerow([t, P, PI, Ax, Ay, p_diff, drag_Lagrangian, S0, lift_Lagrangian, S1])


result = "results_UpdateALE"		# name of folder containing results

# load mesh with boundary and domain markers
import marker
(mesh, bndry, interface, unelastic_surface, domains, A, B) \
        = marker.give_marked_mesh(mesh_coarseness = 50, refine = True, ALE = True)

# domain (used while building mesh) - needed for inflow condition
gW = 0.41

# boundary marks' names (already setted to the mesh) - needed for boundary conditions
_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

dx  = dx(domain=mesh, subdomain_data = domains)
ds  = ds(domain=mesh, subdomain_data = bndry)
dS  = dS(domain=mesh, subdomain_data = interface)
dss = ds(domain=mesh, subdomain_data = unelastic_surface)

# time stepping
dt    = 0.002
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


flow = Flow(mesh, bndry, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result)

t = 0.0

while  t < t_end:
    info("t = %.3f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    flow.save(t)

    t += float(dt)

flow.data.close()
