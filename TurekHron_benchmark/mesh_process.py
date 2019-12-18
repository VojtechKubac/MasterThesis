"""
This code was implemented by Vojtech Kubac as a part of his Master Thesis that will be defended
on February 2020. At the Faculty of Mathematics and Physics of Charles University in Prague.
"""

import sys
import os.path
from dolfin import *
from mshr import *
import numpy as np

parameters["refinement_algorithm"] = "plaza_with_parent_facets"
#parameters['allow_extrapolation'] = True

def read_mesh_xml(name):
    mesh = Mesh(name+'.xml')
    bd=None
    sd=None
    
    if os.path.isfile(name+"_facet_region.xml") : 
        bd = MeshFunction("size_t", mesh, name+"_facet_region.xml")

    if os.path.isfile(name+"_physical_region.xml") : 
        sd = MeshFunction("size_t", mesh, name+"_physical_region.xml")

    mesh.init()
    #plot(bd, interactive=True)
    #plot(sd, interactive=True)
    return(mesh,bd,sd)

# for parsing input arguments
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--mesh", dest="mesh_name", default='bench3D')
(options, args) = parser.parse_args()

name=options.mesh_name

mesh, bd, sd =read_mesh_xml(name)

print("Save the hdf mesh and boundaries...")
hdf = HDF5File(mesh.mpi_comm(), name+".h5", "w")
hdf.write(mesh, "/mesh")
if(sd) : hdf.write(sd, "/domains")
if(bd) : hdf.write(bd, "/bndry")

#save mesh just for paraview if needed
print("Save the xdmf mesh...")    
meshfile = XDMFFile(MPI.comm_world,name+"_pv_mesh.xdmf")
meshfile.write(mesh)
if(bd) :
    bdfile = XDMFFile(MPI.comm_world,name+"_pv_facets.xdmf")
    bdfile.write(bd)
if(sd) :
    sdfile = XDMFFile(MPI.comm_world,name+"_pv_cells.xdmf")
    sdfile.write(sd)

print("Done.")
