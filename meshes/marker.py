from dolfin import *
from dolfin import __version__
import mshr
import os.path

# interface marks
_FSI = 1
_FLUID_CYLINDER = 2

# bndry marker
_CYLINDER = 3

gX = 0.2		# x coordinate of the centre of the circle
gY = 0.2		# y coordinate of the centre of the circle
gEH = 0.02		# hight of the elastic part

A = Point(0.6, 0.2)	# point at the end of elastic beam - for pressure comparison
B = Point(0.15, 0.2)	# point at the surface of rigid circle - for pressure comparison


def give_gmsh_mesh(name):
    if not os.path.isfile(name):
        raise ValueError('{} is an invalid name for gmsh mesh.'.format(name))

    info('processing mesh...')

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)

    dim = mesh.geometry().dim()

    domains = MeshFunction('size_t', mesh, dim, mesh.domains())
    hdf.read(domains, '/domains')
    bndry = MeshFunction('size_t', mesh, dim - 1)
    hdf.read(bndry, '/bndry')    
    interface = MeshFunction('size_t', mesh, dim - 1)
    interface.set_all(0)
    
    for f in facets(mesh):
        if not f.exterior():
            flag = 0
            for c in cells(f):
                if domains[c] == 0:
                    flag |= 1
                if domains[c] == 1:
                    flag |= 2
            if flag == 3:
                interface[f] = _FSI
        else:
            mp = f.midpoint()
            if bndry[f] == _CYLINDER and not(mp[0] > gX and mp[1] < gY + 0.5*gEH + DOLFIN_EPS \
                    and mp[1] > gY - 0.5*gEH - DOLFIN_EPS):
                interface[f] = _FLUID_CYLINDER

    return(mesh, bndry, domains, interface, A, B)
