from dolfin import *
from dolfin import __version__
import mshr
import os.path

# interface marks
_FSI = 1


def give_gmsh_mesh(name):
    if not os.path.isfile(name):
        raise ValueError('{} is an invalid name for gmsh mesh.'.format(name))

    info('processing mesh...')

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)

    dim = mesh.geometry().dim()

    domains = MeshFunction('size_t', mesh, dim, mesh.domains())
    #domains.set_all(0)
    hdf.read(domains, '/domains')
    #hdf.read(domains, '/subdomains')
    bndry = MeshFunction('size_t', mesh, dim - 1)
    #bndry.set_all(0)
    hdf.read(bndry, '/bndry')    
    #hdf.read(bndry, '/boundaries')    
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

    return(mesh, bndry, domains, interface)


if __name__ == '__main__':
    
    for par in [30,40,50,60,70,80]:
        for refinement in [True, False]:
            for ALE in [True, False]:
                name = 'mesh{}'.format(par)
                if refinement:
       	            name += '_refined'
                if ALE:
                    name += '_ALE'
                name += '.h5'
                
                generate_mesh(name, par, refinement, ALE)
