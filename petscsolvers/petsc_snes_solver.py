"""
Author of this PETSc TS interface is Jaroslav Hron, the supervisor of Vojtech Kubac's Master Thesis.
This is a content of a subfolder to the codes used within the Thesis.
"""

import dolfin as df
import numpy as np
from petsc4py import PETSc

# classes are modeled to have simmilar call signature as
#problem = NonlinearVariationalProblem(F, w, bcs, derivative(F,w), form_compiler_parameters=ffc_opts)
#solver = NonlinearVariationalSolver(problem)

#df.set_log_level(df.LogLevel.TRACE)
#df.set_log_level(df.TRACE)

class Problem(object):
    def __init__(self, F, w, bcs, J, *args, **kwargs):
        self.F_form = F
        self.J_form = J
        self.bcs = bcs
        self.w = w

        self.form_compiler_parameters = {"optimize": True}
        if "form_compiler_parameters" in kwargs:
            self.form_compiler_parameters = kwargs["form_compiler_parameters"]

        self.assembler = df.SystemAssembler(self.J_form, self.F_form, self.bcs)

        self.x = df.as_backend_type(w.vector())
        self.x_petsc = self.x.vec()

        self.b_petsc = self.x_petsc.duplicate()

        # We need to initialise the matrix size and local-to-global maps
        self.A_dolfin = df.PETScMatrix()
        self.assembler.init_global_tensor(self.A_dolfin, df.Form(self.J_form))
        self.A_petsc = self.A_dolfin.mat()        
        self.xx=self.A_petsc.createVecRight()
        
    def update_x(self, x):
        """Given a PETSc Vec x, update the storage of our solution function u."""
        x.copy(self.x_petsc)
        self.x.update_ghost_values()

    def jacobian(self, snes, x, A, P):
        """The callback that the SNES executes to assemble our Jacobian."""
        self.update_x(x)
        A_wrap = df.PETScMatrix(A)
        self.assembler.assemble(A_wrap)
        
    def residual(self, snes, x, b):
        """The callback that the SNES executes to compute the residual."""
        self.update_x(x)
        b_wrap = df.PETScVector(b)
        self.assembler.assemble(b_wrap, self.x)

class Solver(object):
    def __init__(self, problem, tag, *args, **kwargs):
        self.problem = problem
        self.tag=tag
        opts = PETSc.Options()

        self.snes = PETSc.SNES().create()
        snes=self.snes
        snes.setFunction(problem.residual, problem.b_petsc)
        snes.setJacobian(problem.jacobian, problem.A_petsc)
        snes.setType("newtonls")

        opts['snes_linesearch_type']='basic'
        snes.setTolerances(rtol=1e-10, atol=1e-10, stol=1e-10, max_it=20)
        snes.computeJacobian(problem.xx, problem.A_petsc)
        snes.setMonitor(SNESMonitor(self))

        ksp = self.snes.getKSP()
        ksp.setType('fgmres') #('preonly')
        ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=400)
        ksp.setGMRESRestart(400)
        #opts['ksp_gmres_restart']=400
        #opts['ksp_monitor']=''
        #opts['ksp_converged_reason']=''
        #opts['ksp_monitor_max']=''
        ksp.setMonitor(KSPMonitor(self))
                                                
        pc = ksp.getPC()        
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')
        opts['mat_mumps_icntl_14']= 400
        opts['mat_mumps_cntl_1']= 1e-6
                
    def solve(self) :
        self.snes.setFromOptions()
        snes.setUp()
        snes.view()
        self.snes.solve(None, self.problem.xx)
        self.problem.update_x(self.problem.xx)
        its=self.snes.getIterationNumber()
        ok=self.snes.converged
        return(its, ok)
        
