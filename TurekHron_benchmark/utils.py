"""
This code was implemented by Vojtech Kubac as a part of his Master Thesis that will be defended
on February 2020. At the Faculty of Mathematics and Physics of Charles University in Prague.
"""

from dolfin import *
from dolfin import __version__
import mshr
import os.path

# mesh specific constants
gW = 0.41		# width of the domain
gL = 2.5		# length of the domain
gX = 0.2		# x coordinate of the centre of the circle
gY = 0.2		# y coordinate of the centre of the circle
g_radius  = 0.05	# radius of the circle
gEL = 0.35		# length of the elastic part (left end fully attached to the circle)
gEH = 0.02		# hight of the elastic part

A = Point(0.6, 0.2)	# point at the end of elastic beam - for pressure comparison
B = Point(0.15, 0.2)	# point at the surface of rigid circle - for pressure comparison

# boundary marks
_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

# interface marks
_FSI = 1
_FLUID_CYLINDER = 2


def give_gmsh_mesh(name):
    if not os.path.isfile(name):
        raise ValueError('Invalid name for gmsh mesh.')

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name, 'r')
    hdf.read(mesh, '/mesh', False)
    domains = MeshFunction('size_t', mesh, 2, mesh.domains())
    hdf.read(domains, '/domains')
    bndry = MeshFunction('size_t', mesh, 1)
    hdf.read(bndry, '/bndry')    
    interface = MeshFunction('size_t', mesh, 1)
    interface.set_all(0)
    
    for f in facets(mesh):
        if f.exterior():
            mp = f.midpoint()
            if bndry[f] == _CIRCLE and not(mp[0] > gX and mp[1] < gY + gEH + DOLFIN_EPS \
                    and mp[1] > gY - gEH - DOLFIN_EPS):
                interface[f] = _FLUID_CYLINDER
        else:
            flag = 0
            for c in cells(f):
                if domains[c] == 0:
                    flag |= 1
                if domains[c] == 1:
                    flag |= 2
            if flag == 3:
                interface[f] = _FSI

    return(mesh, bndry, domains, interface, A, B)


def get_benchmark_specification(benchmark = 'FSI1'):
    if benchmark == 'FSI1':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 0.2
        Ae = 3.5e04
        T_end = 60.0
        result = "results-FSI1"		
        dt = 0.02
    elif benchmark == 'FSI2':
        rho_s = Constant(1e04)
        nu_s = Constant(0.4)
        mu_s = Constant(5e05)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 1.0
        Ae = 1.4e03
        T_end = 35.0
        result = "results-FSI2"		
        dt = 0.002   # 0.001 
    elif benchmark == 'FSI3':
        rho_s = Constant(1e03)
        nu_s = Constant(0.4)
        mu_s = Constant(2e06)
        rho_f = Constant(1e03)
        nu_f = Constant(1e-03)
        U = 2.0
        Ae = 1.4e03
        T_end = 20.0
        result = "results-FSI3"		
        dt = 0.0005 # 0.001 # 0.002 #
    else:
        raise ValueError('"{}" is a wrong name for problem specification.'.format(benchmark))
    v_max = Constant(1.5*U)     # mean velocity to maximum velocity 
                                #      (we have parabolic profile)
    E_s = Constant(Ae*rho_f*U*U)
    lambda_s = Constant((nu_s*E_s)/((1+nu_s)*(1-2*nu_s)))
    mu_f = Constant(nu_f*rho_f)
    return v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, T_end, dt, result
