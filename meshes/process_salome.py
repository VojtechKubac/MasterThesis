#!/usr/bin/env python

import sys
import os.path
from dolfin import *
from mshr import *
import numpy as np
from dolfin import __version__

if __version__[:4] == '2018' or  __version__[:4] == '2019':
    comm = MPI.comm_world
else:
    comm = mpi_comm_world()

parameters["refinement_algorithm"] = "plaza_with_parent_facets"
#parameters['allow_extrapolation'] = True

# for parsing input arguments
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--mesh", dest="mesh_name", default='bench_bent_tube')
(options, args) = parser.parse_args()

name=options.mesh_name

mesh = Mesh(name+'.xml')

mesh.init()

# boundary marks' names (already setted to the mesh) - needed for boundary conditions 
_INFLOW      = 1 
_INFLOW_SOL  = 2 
_OUTFLOW     = 3 
_OUTFLOW_SOL = 4
_WALL        = 5
_FSI         = 6

subdomains = MeshFunction('size_t', mesh, 3, mesh.domains())
boundary = MeshFunction('size_t', mesh, 2)
subdomains.set_all(0)
boundary.set_all(0)

# mark subdomains
class Solid(SubDomain):
    #def snap(self, x):
    #    r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
    #    if r <= g_radius:
    #        x[0] = gX + (g_radius/r) * (x[0] - gX)
    #        x[1] = gY + (g_radius/r) * (x[1] - gY)
    def inside(self, x, on_boundary):
        if x[1] < 2.0 + DOLFIN_EPS:
            return x[0]**2 + x[2]**2 > 1.0 - DOLFIN_EPS
        elif x[0] > 5.0 - DOLFIN_EPS:
            return  np.sqrt((x[1] - 7.0)**2 + x[2]**2) > 1.0 - DOLFIN_EPS*1e3
        else:
            d1 = x[2]**2
            d2 = (sqrt((x[0] - 5.0)**2 + (x[1] - 2.0)**2) - 5.0)**2
            return np.sqrt(d1 + d2) > 1.0 - DOLFIN_EPS*1e3

        return False

solid = Solid()
solid.mark(subdomains, 1)

zero_point = Point((0, 0, 0))
point2 = Point((7, 7, 0))
# mark FSI interace and boundary
for f in facets(mesh):
    if f.exterior():
        mp = f.midpoint()
        if mp[1] < DOLFIN_EPS:  # inflow
            if mp.distance(zero_point) < 1.0 + DOLFIN_EPS:
                boundary[f] = _INFLOW
            else:
                boundary[f] = _INFLOW_SOL
        elif mp[0] > 7.0 - DOLFIN_EPS*1e3:
            if mp.distance(point2) < 1.0 + DOLFIN_EPS:
                boundary[f] = _OUTFLOW
            else:
                boundary[f] = _OUTFLOW_SOL
        else: boundary[f] = _WALL

    flag = 0
    for c in cells(f):
        if subdomains[c] == 0:
            flag |= 1
        if subdomains[c] == 1:
            flag |= 2
    if flag == 3:
        boundary[f] = _FSI


print("Save the hdf mesh and boundaries...")
hdf = HDF5File(mesh.mpi_comm(), name+"_L0.h5", "w")
hdf.write(mesh, "/mesh")
if(subdomains) : hdf.write(subdomains, "/domains")
if(boundary) : hdf.write(boundary, "/bndry")

#save mesh just for paraview if needed
print("Save the xdmf mesh...")
meshfile = XDMFFile(comm,name+"_L0_pv_mesh.xdmf")
meshfile.write(mesh)

if(boundary) :
    bdfile = XDMFFile(comm,name+"_L0_pv_facets.xdmf")
    bdfile.write(boundary)
if(subdomains) :
    subdomainsfile = XDMFFile(comm,name+"_L0_pv_cells.xdmf")
    subdomainsfile.write(subdomains)

print("Done.")
