import dolfin as df
import numpy as np
from petsc4py import PETSc
import sys
import os.path
#sys.path.append('../../petscsolvers/')
from petsc_monitors import TSMonitor, SNESMonitor, KSPMonitor

#df.set_log_level(df.LogLevel.TRACE) # 2018.1.0
#df.set_log_level(df.TRACE)          # 2017.2.0 
print=PETSc.Sys.Print

class Problem(object):
    def __init__(self, F_mesh, F_physical, w, wdot, bcs_mesh, bcs, dF_mesh_u, dF_mesh_udot, 
             dF_physical_u, dF_physical_udot, update=None, report=None, *args, **kwargs):
        self.F_mesh_form = F_mesh
        self.dF_mesh_u_form = dF_mesh_u
        self.dF_mesh_udot_form = dF_mesh_udot
        self.F_physical_form = F_physical
        self.dF_physical_u_form = dF_physical_u
        self.dF_physical_udot_form = dF_physical_udot
        self.bcs_mesh = bcs_mesh
        self.bcs = bcs
        self.w = w
        self.wdot = wdot
        self.a = df.Constant(1.0)
        self.dF_mesh_form = self.dF_mesh_u_form + self.a * self.dF_mesh_udot_form
        self.dF_physical_form = self.dF_physical_u_form + self.a * self.dF_physical_udot_form

        self.report=report
        self.update=update
        
        self.form_compiler_parameters = {"optimize": True}
        if "form_compiler_parameters" in kwargs:
            self.form_compiler_parameters = kwargs["form_compiler_parameters"]

        self.assembler = df.SystemAssembler(self.dF_mesh_form + self.dF_physical_form,
                 self.F_mesh_form + self.F_physical_form, self.bcs + self.bcs_mesh, form_compiler_parameters=self.form_compiler_parameters)

        self.x = df.as_backend_type(w.vector())
        self.x_petsc = self.x.vec()
        self.xdot = df.as_backend_type(wdot.vector())
        self.xdot_petsc = self.xdot.vec()

        # We need to initialise the matrix size and local-to-global maps
        self.A_dolfin = df.PETScMatrix()
        self.assembler.init_global_tensor(self.A_dolfin, df.Form(self.dF_mesh_form+self.dF_mesh_form))
        self.A_petsc = self.A_dolfin.mat()
        self.xx=self.A_petsc.createVecRight()
        self.xx.axpy(1.0,self.x_petsc)
        self.xxdot=self.A_petsc.createVecRight()

        self.b_petsc = self.A_petsc.createVecLeft()

        
        self.A1_dolfin = df.PETScMatrix()
        self.assembler.init_global_tensor(self.A1_dolfin, df.Form(self.dF_mesh_form+self.dF_mesh_form))
        self.A2_dolfin = df.PETScMatrix()
        self.assembler.init_global_tensor(self.A2_dolfin, df.Form(self.dF_mesh_form+self.dF_mesh_form))

    def update_x(self, x):
        """Given a PETSc Vec x, update the storage of our solution function u."""
        x.copy(self.x_petsc)
        self.x.update_ghost_values()

    def update_xdot(self, xdot):
        """Given a PETSc Vec xdot, update the storage of our solution function u."""
        xdot.copy(self.xdot_petsc)
        self.xdot.update_ghost_values()

    def evalFunction(self, ts, t, x, xdot, b):
        """The callback that the TS executes to compute the residual."""
        self.update(t)
        self.update_x(x)
        self.update_xdot(xdot)

        b1 = df.Vector()
        b2 = df.Vector()
        #self.assembler.assemble(self.F_mesh_form, tensor = b1)
        df.assemble(self.F_mesh_form, tensor = b1)
        [bc.apply(b1) for bc in self.bcs_mesh]

        #self.assembler.assemble(self.F_physical_form, tensor = b2)
        df.assemble(self.F_physical_form, tensor = b2)
                                                        
        b_wrap = df.PETScVector(b)
        # zero all entries
        b_wrap.zero()
        #df.assemble(b_wrap, self.x)

        b_wrap.axpy(1,b1)
        b_wrap.axpy(1,b2)
        [bc.apply(b_wrap, self.x) for bc in self.bcs]

        #df.assemble(self.F_form, tensor=b_wrap, form_compiler_parameters=self.form_compiler_parameters)
        #for bc in self.bcs : bc.apply(b_wrap, self.x)
                
    def evalJacobian(self, ts, t, x, xdot, a, A, P):
        """The callback that the TS executes to compute the jacobian."""
        self.update(t)
        self.update_x(x)
        self.update_xdot(xdot)
        self.a.assign(a)
        A_wrap = df.PETScMatrix(A)

        # set all entries to zero
        A_wrap.zero()

        A_wrap.apply('insert')

        #self.assembler.assemble(self.dF_mesh, tensor = self.A1_dolfin, keep_diagonal=True)
        df.assemble(self.dF_mesh_form, tensor = self.A1_dolfin, keep_diagonal=True)
        [bc.zero(self.A1_dolfin) for bc in self.bcs_mesh]
        #self.assembler.ssemble(self.dF, tensor = self.A2_dolfin, keep_diagonal=True)
        df.assemble(self.dF_physical_form, tensor = self.A2_dolfin, keep_diagonal=True)

        A_wrap.axpy(1, self.A1_dolfin, False)
        A_wrap.axpy(1, self.A2_dolfin, False)
        [bc.apply(A_wrap) for bc in self.bcs]

        #df.assemble(self.J_form, tensor=A_wrap, form_compiler_parameters=self.form_compiler_parameters)
        #for bc in self.bcs : bc.apply(A_wrap)
        #print("Mat info:", A.getInfo())
        return True # same nonzero pattern
        
class Solver(object):
    def __init__(self, problem, tag, *args, **kwargs):
        self.problem = problem
        self.tag=tag
        opts = PETSc.Options()

        # Timestepping solver
        self.ts = PETSc.TS().create()
        self.ts_its=0
        
        ts=self.ts
        ts.setIFunction(problem.evalFunction, problem.b_petsc)
        ts.setIJacobian(problem.evalJacobian, problem.A_petsc)
        ts.setSolution(problem.xx)
        ts.computeIFunction(0.0, problem.xx, problem.xxdot, problem.b_petsc)
        ts.computeIJacobian(0.0, problem.xx, problem.xxdot, 1.0, problem.A_petsc)
        ts.setMonitor(TSMonitor(self))

        # Nonlinear solver
        self.snes = ts.getSNES()
        
        self.snes_its=0
        self.snes.setMonitor(SNESMonitor(self))

        # Linear solver
        self.ksp = self.snes.getKSP()
        self.ksp_its=0
        self.ksp.setMonitor(KSPMonitor(self))

        # and preconditioner
        self.pc = self.ksp.getPC()        

        ts.setFromOptions()
        #ts.setUp()
        #ts.view()

    def solve(self) :
        ts=self.ts
        ts.setFromOptions()
        #ts.setUp()
        #ts.view()
        ts.solve(self.problem.xx)
        self.problem.update_x(self.problem.xx)
        its=ts.getStepNumber()
        ok=ts.converged
        return(its,ok)
