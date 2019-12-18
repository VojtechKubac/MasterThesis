"""
Author of this PETSc TS interface is Jaroslav Hron, the supervisor of Vojtech Kubac's Master Thesis.
This is a content of a subfolder to the codes used within the Thesis.
"""

from petsc4py import PETSc
import time
import dolfin as df

### report class using petsc print
class Report(object):
    def __init__(self, max_level=99):
        self.max_level=max_level
        self.level=0
        self.time_stamp=[time.perf_counter()]
        self.index=[0]
        self.names=[""]
        self.name=None
        
    def __call__(self,s):
        self.name=s
        return(self)
    
    def __enter__(self):
        self.begin(self.name)
        return

    def __exit__(self, type, value, traceback):
        self.end()
        self.name=None
        return

    def print(self, i, name=None, mode="-", str='', end="\n", *args, **kwargs):
        if name is None : name=""
        if i<0 :
            PETSc.Sys.Print(str, sep="", end=end, *args, **kwargs)
        else :
            t=time.perf_counter()-self.time_stamp[0]
            PETSc.Sys.Print("{0:1d}".format(i)+ "|" + " {0:>10f} ".format(t) + ("_"*(self.level-1)) + mode + "{0:>6s}| ".format(name) + str, sep="", end=end, *args, **kwargs)
                    
    def begin(self,name):
        self.time_stamp.append(time.perf_counter())
        self.level +=1
        self.index.append(self.level)
        self.names.append(name)
        i=self.index[-1]
        if self.level<self.max_level: self.print(i, name, ">", str="start...")
        elif self.level==self.max_level: self.print(i, name, ">", str="start...", end="")
        

    def log(self,s, mode="."):
        i=self.index[-1]
        name=self.names[-1]
        if self.level<self.max_level: self.print(i,name, mode=mode, str=s)
        elif self.level==self.max_level: self.print(-1,name, mode=mode, str=".", end="")
            
    def end(self,s=""):
        i=self.index.pop()
        name=self.names.pop()
        now=time.perf_counter()
        last=self.time_stamp.pop()
        if self.level<self.max_level: self.print(i, name, "<", str="done. " + s + " [ {0:.2e}s ]".format(now-last))
        elif self.level==self.max_level: self.print(-1, name, "<", str="done. " + s + " [ {0:.2e}s ]".format(now-last))
        self.level = self.level-1

    def info(self,s):
        i=self.index[-1]
        self.print(i, name=None, str=s)
        
    def time(self):
        self.print(self.index[-1], name="TIME", mode="!", str=time.strftime("[%a, %d %b %Y %H:%M:%S]", time.gmtime()))
        
