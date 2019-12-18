"""
Author of this PETSc TS interface is Jaroslav Hron, the supervisor of Vojtech Kubac's Master Thesis.
This is a content of a subfolder to the codes used within the Thesis.
"""

import dolfin as df
import numpy as np
from petsc4py import PETSc

# supply own monitors for now

class KSPMonitor(object):
    def __init__(self, solver, name='KSP'):
        self.solver=solver
        self.tag=solver.tag
        self.name=name
        
    def __call__(self, ksp, its, rnorm, *args, **kwargs):
        if (its==0):
            self.rnorm0 = rnorm
            self.tag.begin(self.name)
        else:
            if ksp.iterating :
                if (its==1) :
                    s=('%6s' % "it") + (' %12s' % "l2-norm") + (' %12s' % "rel l2-norm") + (' %12s' % "l1-norm") + (' %12s' % "inf-norm")+ (' %12s' % "energy-norm")
                    self.tag.log(s)
                r=ksp.buildResidual()
                rn1 = r.norm(PETSc.NormType.NORM_1)
                rn2 = r.norm(PETSc.NormType.NORM_2)
                rni = r.norm(PETSc.NormType.NORM_INFINITY)
                rne = r.dot(ksp.vec_sol)
                s = ('%6d' % its) + (' %12.2e' % rnorm) + (' %12.2e' % (rnorm/self.rnorm0))+ (' %12.2e' % (rn1))+ (' %12.2e' % (rni))+ (' %12.2e' % (rne))
                self.tag.log(s)
            else:
                #self.tag.log("Result: {0}".format(ksp.reason))
                self.solver.ksp_its=its
                self.tag.end()
            
class SNESMonitor(object):
    def __init__(self, solver, name='SNES'):
        self.solver=solver
        self.tag=solver.tag
        self.name=name
        self.active=False
        
    def __call__(self, snes, its, rnorm, *args, **kwargs):
        if not self.active:
            self.rnorm0 = rnorm if rnorm>0 else 1.0
            s=('%6s' % "it") + (' %12s' % "l2-norm") + (' %12s' % "rel l2-norm") + (' %12s' % "l1-norm") + (' %12s' % "inf-norm")+ (' %12s' % "energy-norm")
            self.tag.begin(self.name)
            self.tag.log(s, mode="#")
            self.active=True

        xnorm=snes.vec_sol.norm(PETSc.NormType.NORM_2)
        ynorm=snes.vec_upd.norm(PETSc.NormType.NORM_2)
        iterating = (snes.callConvergenceTest(its, xnorm, ynorm, snes.norm) == 0)

        (f,_)=snes.getFunction()
        fn1 = f.norm(PETSc.NormType.NORM_1)
        fn2 = f.norm(PETSc.NormType.NORM_2)
        fni = f.norm(PETSc.NormType.NORM_INFINITY)
        fne = f.dot(snes.vec_sol)
        s = ('%6d' % its) + (' %12.2e' % rnorm) + (' %12.2e' % (rnorm/self.rnorm0))+ (' %12.2e' % (fn1))+ (' %12.2e' % (fni))+ (' %12.2e' % (fne))
        self.tag.log(s)
        if not iterating :
            #self.tag.log("Result: {0}".format(snes.reason))
            self.solver.snes_its=its
            self.active=False
            self.tag.end(" {} iterations ".format(its))

class TSMonitor(object):
    def __init__(self, solver, name='TS'):
        self.solver=solver
        self.tag=solver.tag
        self.name=name
        self.report=solver.problem.report
        self.active=False
        
    def __call__(self, ts, its, t, x, *args, **kwargs):
        dt = ts.time_step
        nu  = ts.vec_sol.norm()
        if self.report is not None: self.report(t)
        if not self.active:
            s=('%6s' % 'its') + (' %12s' % 'time') + (' %12s' % 'dt')+ (' %12s' % 'norm(u)') + (' %4s' % 'SNES') + (' %4s' % 'KSP' )
            self.tag.begin(self.name)
            self.active=True
            self.tag.log(s)
            
        s=('%6d' % its) + (' %12.2e' % t) + (' %12.2e' % dt)+ (' %12.2e' % (nu)) + (' %4d' % self.solver.snes_its) + (' %4d' % self.solver.ksp_its )
        self.tag.log(s)
        if not ts.iterating:
            self.solver.ts_its=its
            self.active=False
            self.tag.end()
