"""
This code was implemented by Vojtech Kubac as a part of his Master Thesis that will be defended
on February 2020. At the Faculty of Mathematics and Physics of Charles University in Prague.

The C++ code is thanks to Jan Blechta.
"""

from dolfin import compile_cpp_code

_cpp = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <dolfin/fem/UFC.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>

namespace py = pybind11;

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("assemble_local_cutcell",
        [](Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A,
           const dolfin::Form& a,
           const dolfin::Cell& cell,
           //dolfin::UFC& ufc,
           //const std::vector<double>& coordinate_dofs,
           //ufc::cell& ufc_cell, ///< [in]
           Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& quadrature_points,
           Eigen::Matrix<double, Eigen::Dynamic, 1>& quadrature_weights)
        {
          dolfin::UFC ufc(a);
          ufc::cell ufc_cell;
          std::vector<double> coordinate_dofs;

          std::size_t N, M;
          if (a.rank() == 0)
          {
            N = 1;
            M = 1;
          }
          else if (a.rank() == 1)
          {
            N = a.function_space(0)->dofmap()->cell_dofs(cell.index()).size();
            M = 1;
          }
          else
          {
            N = a.function_space(0)->dofmap()->cell_dofs(cell.index()).size();
            M = a.function_space(1)->dofmap()->cell_dofs(cell.index()).size();
          }
          A.resize(N, M);

          cell.get_cell_data(ufc_cell);
          cell.get_coordinate_dofs(coordinate_dofs);

          ufc::cutcell_integral* integral = ufc.default_cutcell_integral.get();

          //ufc.update(cell, coordinate_dofs, ufc_cell);
          ufc.update(cell, coordinate_dofs, ufc_cell, integral->enabled_coefficients());

          A.setZero();
          integral->tabulate_tensor(A.data(),
                                    ufc.w(),
                                    coordinate_dofs.data(),
                                    quadrature_weights.size(),
                                    quadrature_points.data(),  // FIXME: correct datatype?!?
                                    quadrature_weights.data(),
                                    ufc_cell.orientation);
        });
}
"""
assemble_local_cutcell = compile_cpp_code(_cpp).assemble_local_cutcell

from dolfin.fem.assembling import _create_dolfin_form
from dolfin import *
from ffc.fiatinterface import _create_fiat_element
import numpy as np
import FIAT

comm = MPI.comm_world


# create fiat el
edges = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
faces = {0: (0, 1, 2)}
topology = {0: {0: (0,), 1: (1,), 2: (2,)}, 1: edges, 2: faces}


def bisection(A, B, flag_A, ips):
    """
    Called by 'find_division_point'.

    Finds the point of intersection of given edge and discontinuity interface.
    """
    TOL = 1e-12
    M = 0.5*(A + B)
    if np.linalg.norm(A - B) < TOL:
        return(M)
    else:
        TOL2 = DOLFIN_EPS
        flag_M = ips(M)
        if flag_M == flag_A:
            return(bisection(M, B, flag_A, ips))
        else:
            return(bisection(A, M, flag_A, ips))


def find_division_points(flags, points, ips):
    """
    Called by 'assemble_cutcell'.

    Finds points (new_point0, new_point1) where interface intersects edges of simplex.
    The ordering of new points is counter-clockwise, starting from points[0].
    Also returnsnumber of vertex whose adjacent edges are intersected.
    """

    if flags[0] == flags[1]:
        vertex = 2
        new_point0 = bisection(points[1], points[2], flags[1], ips)
        new_point1 = bisection(points[0], points[2], flags[0], ips)
    elif flags[1] == flags[2]:
        vertex = 0
        new_point0 = bisection(points[0], points[1], flags[0], ips)
        new_point1 = bisection(points[0], points[2], flags[0], ips)
    else:
        vertex = 1
        new_point0 = bisection(points[0], points[1], flags[0], ips)
        new_point1 = bisection(points[1], points[2], flags[1], ips)
    #info("new_point0 = {}, new_point1 = {}".format(new_point0, new_point1))
    return(vertex, new_point0, new_point1)


def integrate_subcell(points, a, c):
    """
    Performs integration of 'a_dolfin' over given cell 'c' within the smaller triangle specified
    by 'points'.

    Called by 'vertex_edge_integration' or 'edge_edge_integration'.

    It returns the assembled matrix 'A'.
    """
    #info('integrate subcell')
    #print(points)
    #print(type(a))
    if a.rank() == 0:
        N = 1
        M = 1
    elif a.rank() == 1:
        #N = a.function_space(0).dofmap().cell_dofs(c.index()).size()
        N = a.function_space(0).dofmap().cell_dofs(c.index()).size
        M = 1
    else:
        #N = a.function_space(0).dofmap().cell_dofs(c.index()).size()
        #M = a.function_space(0).dofmap().cell_dofs(c.index()).size()
        N = a.function_space(0).dofmap().cell_dofs(c.index()).size
        M = a.function_space(0).dofmap().cell_dofs(c.index()).size

    A = np.zeros((N, M))
    verts = (points[0], points[1], points[2])
    el=FIAT.reference_element.UFCSimplex(FIAT.reference_element.TRIANGLE, verts, topology)
    scheme = FIAT.quadrature_schemes.create_quadrature(el, 3)
    pts = scheme.get_points()  # FIXME: transform to physical coords?!?  YES
    wts = scheme.get_weights()
    #info('assemble in integration_cutcell')
    assemble_local_cutcell(A, a, c, pts, wts)
    #print('integrate subcell: ', A.shape)
    return A

def integrate_interface(A, B, c, F_fluid_x, F_solid_x, F_fluid_y, F_solid_y, ips):
    """
    Surface integration over interface.
    Points A, B are the intersection point and w(=Dv) is derivative in the direction of v.
    """
    #if F_fluid.rank() == 0:
    #    N = 1
    #    M = 1
    #elif a.rank() == 1:
    #    #N = a.function_space(0).dofmap().cell_dofs(c.index()).size()
    #    N = a.function_space(0).dofmap().cell_dofs(c.index()).size
    #    M = 1
    #else:
    #    #N = a.function_space(0).dofmap().cell_dofs(c.index()).size()
    #    #M = a.function_space(0).dofmap().cell_dofs(c.index()).size()
    #    N = a.function_space(0).dofmap().cell_dofs(c.index()).size
    #    M = a.function_space(0).dofmap().cell_dofs(c.index()).size
    #info('integrate interface')

    t = A - B
    #info('compute distance')
    dist = np.linalg.norm(t)
    n = np.array([t[1], -t[0]])/dist
    #info('create expression')
    #n_exp = Expression(("n0", "n1"), degree=0, n0=0, n1=-1)

    #info('define normals')
    if ips(A + n) == 1:
        n_s = n#_exp
        n_f = -n#_exp
    else:
        n_s = -n#_exp
        n_f = n#_exp

    pts = np.array([A, B])
    wts = np.array([0.5, 0.5])*dist
    """
    #info('assemble in integration_cutcell')
    info('create form')
    #a = _create_dolfin_form(dot(w, n_f)*F_fluid*dC + dot(w, n_s)*F_solid*dC)
    #a = _create_dolfin_form(F_fluid*dC + F_solid*dC)
    a = F_fluid

    N_ = a.function_space(0).dofmap().cell_dofs(c.index()).size
    M_ = a.function_space(0).dofmap().cell_dofs(c.index()).size

    MAT = np.zeros((N_, M_))

    info('assemble')
    assemble_local_cutcell(MAT, a, c, pts, wts)
    info('{}'.format(MAT))
    info('redifine N')
    F_fluid.N.a = 0
    F_fluid.N.b = 0
    info('assemble2')
    assemble_local_cutcell(MAT, a, c, pts, wts)
    info('{}'.format(MAT))
    #print('integrate subcell: ', A.shape)
    info('end of integrate interface')
    return MAT
    """
    weight = 0.5/dist

    a = F_fluid_x

    N = a.function_space(0).dofmap().cell_dofs(c.index()).size
    M = a.function_space(0).dofmap().cell_dofs(c.index()).size

    MAT = np.zeros((N, M))

    #info('assemble')
    assemble_local_cutcell(MAT, a, c, pts, wts)
    MAT_final = n_f[0]*MAT

    MAT = np.zeros((N, M))
    assemble_local_cutcell(MAT, F_fluid_y, c, pts, wts)
    MAT_final += n_f[1]*MAT

    MAT = np.zeros((N, M))
    assemble_local_cutcell(MAT, F_solid_x, c, pts, wts)
    MAT_final += n_s[0]*MAT

    MAT = np.zeros((N, M))
    assemble_local_cutcell(MAT, F_solid_y, c, pts, wts)
    MAT_final += n_s[1]*MAT


    #info('end of integrate interface')
    return MAT_final


def vertex_edge_integration(F_fluid, F_solid, vertex, flags, points, intersection, c):
    """
    Performs cutcell integration for a cell intersected in a vertex and the opposite edge.
    Called from 'divide_and_integrate'.

    In input obtaions cutcell forms for fluid and solid, number of vertex whose edges are intersected,
    vertices of given cell ('points') and whether they are in fluid or solid domain ('flags'),
    coordinates of point of intersection ('intersection') and the cell 'c' itself.

    Devides triangle into two smaller ones (along the line vertex-intersection) and 
    integrates corresponding forms on each triangle. It returns matrix 'A_loc', assembled
    by this procedure.
    """
    ###
    ###   nepotrebuju trojuhelnik rozdelit tak, aby mi nevnikl velky uhel
    ###   jinak bych delil na ctyri trojuhelniky
    ###

    #A_loc1 = np.zeros((local_dim,local_dim))
    #A_loc2 = np.zeros((local_dim,local_dim))
    #A_loc  = np.zeros((local_dim,local_dim))

    #info("Vertex-Edge integration.")
    #print(vertex, points, intersection)

    if vertex == 0:
        verts1 = (points[0], points[1], intersection)
        verts2 = (points[0], intersection, points[2])
        if flags[1] == 0:
            A_loc1 = integrate_subcell(verts1, F_fluid, c)
            A_loc2 = integrate_subcell(verts2, F_solid, c)
        else: 
            A_loc1 = integrate_subcell(verts1, F_solid, c)
            A_loc2 = integrate_subcell(verts2, F_fluid, c)

    elif vertex == 1:
        verts1 = (points[0], points[1], intersection)
        verts2 = (intersection, points[1], points[2])
        if flags[0] == 0:
            A_loc1 = integrate_subcell(verts1, F_fluid, c)
            A_loc2 = integrate_subcell(verts2, F_solid, c)
        else: 
            A_loc1 = integrate_subcell(verts1, F_solid, c)
            A_loc2 = integrate_subcell(verts2, F_fluid, c)

    else:
        verts1 = (points[0], intersection, points[2])
        verts2 = (intersection, points[1], points[2])
        if flags[0] == 0:
            A_loc1 = integrate_subcell(verts1, F_fluid, c)
            A_loc2 = integrate_subcell(verts2, F_solid, c)
        else: 
            A_loc1 = integrate_subcell(verts1, F_solid, c)
            A_loc2 = integrate_subcell(verts2, F_fluid, c)

    A_loc = A_loc1 + A_loc2
    #print('vertex-edge integration: ', A_loc.shape)
    return A_loc

def edge_edge_integration(F_fluid, F_solid, vertex, flags, points, new_point0, new_point1, c):
    """
    Performs cutcell integration for a cell intersected inside two of its edges.
    Called from 'divide_and_integrate'.

    In input obtaions cutcell forms for fluid and solid, number of vertex whose edges are intersected,
    vertices of given cell ('points') and whether they are in fluid or solid domain ('flags'),
    coordinates of points of intersection ('new_point0' and 'new_point1') and the cell 'c' itself.

    Devides triangle into four smaller ones (nonintersected edge is devided in the middle) and 
    integrates corresponding forms on each triangle. It returns matrix 'A_loc', assembled
    by this procedure.
    """

    #info("Edge-Edge integration. \t {} \t {}".format(vertex, points))

    if flags[vertex] == 0:
        a1 = F_fluid
        a2 = F_solid
    else:
        a2 = F_fluid
        a1 = F_solid

    if vertex == 0:
        verts = (points[0], new_point0, new_point1)
        A_loc = integrate_subcell(verts, a1, c)

        new_point2 = 0.5*(points[1] + points[2])
        #info('{}, {}, {}, {}'.format(points, new_point0, new_point1, new_point2))
        verts_list = ((new_point0, points[1], new_point2),
                      (new_point1, new_point2, points[2]),
                      (new_point0, new_point2, new_point1))
        for verts in verts_list:
            A_loc_temp = integrate_subcell(verts, a2, c)
            A_loc = A_loc + A_loc_temp

    elif vertex == 1: 
        verts = (new_point0, points[1], new_point1)
        A_loc = integrate_subcell(verts, a1, c)

        new_point2 = 0.5*(points[0] + points[2])
        #info('{}, {}, {}, {}'.format(points, new_point0, new_point1, new_point2))
        verts_list = ((new_point1, points[2], new_point2),
                      (points[0], new_point0, new_point2),
                      (new_point0, new_point1, new_point2))
        for verts in verts_list:
            A_loc_temp = integrate_subcell(verts, a2, c)
            A_loc = A_loc + A_loc_temp

    else: 
        verts = (new_point1, new_point0, points[2])
        #info('{}'.format(verts))
        A_loc = integrate_subcell(verts, a1, c)

        new_point2 = 0.5*(points[0] + points[1])
        #info('{}, {}, {}, {}'.format(points, new_point0, new_point1, new_point2))
        verts_list = ((points[0], new_point2, new_point1),
                      (new_point2, points[1], new_point0),
                      (new_point0, new_point1, new_point2))
        for verts in verts_list:
            A_loc_temp = integrate_subcell(verts, a2, c)
            A_loc = A_loc + A_loc_temp

    #print('edge-edge integration: ', A_loc.shape)
    return A_loc




def divide_and_integrate(F_fluid, F_fluid_cutcell, F_solid, F_solid_cutcell, zero_form,
                         vertex, flags, points, new_point0, new_point1, c, ips,
                         F_fluid_interface_x=None, F_solid_interface_x=None, 
                         F_fluid_interface_y=None, F_solid_interface_y=None):
    """
    Performs integration over cell on the FSI-interface. Called from 'assemble_cutcell'
    or 'assemble_cutcell_time_dependent'.

    On the input it receives forms for fluid and solid problem, index of vertex whose both edges
    are intersected with the FSI-interface ('vertex'),
    whether they are in the solid or in the fluid part ('flags' 0=fluid, 1=solid), coordinates 
    of vertices of given cell ('points'), 
    coordinates of points of intersection of an edge and FSI-interface ('new_point0' and 'new_point1'), 
    and finally 'c', the cell itself.

    It finds vertex with both edges intersected and decides whether to integrate it as 
    non-intersected (intersection goes close to one edge or just a small piece of cell is intersected
    - uses FEniCS assemble_local routine for that), or whether to integrate is as cutcell
    ('edge_edge_integration' or 'vertex_edge_integration', both defined above).
    """
    #info('divide and integrate')
    #A = np.zeros((local_dim,local_dim))
    test = 0
    if test:
        return assemble_local(zero_form, c)

    closeness_TOL = 1e-12
    cut = 0

    if vertex == 0:
        if np.linalg.norm(points[0] - new_point0) < closeness_TOL:
            if flags[1] == 0:
                #A = assemble_local(F_fluid, c)
                A = assemble_local(zero_form, c)
            else:
                #A = assemble_local(F_solid, c)
                A = assemble_local(F_solid, c)		# solid = solid - fluid
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[0], new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(points[0] - new_point1) < closeness_TOL:
            if flags[2] == 0:
                #A = assemble_local(F_fluid, c)
                A = assemble_local(zero_form, c)
            else:
                #A = assemble_local(F_solid, c)
                A = assemble_local(F_solid, c)		# solid = solid - fluid
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[0], new_point0, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(new_point0 - points[1]) < closeness_TOL:
            if np.linalg.norm(new_point1 - points[2]) < closeness_TOL:
                if flags[0] == 0:
                    A = assemble_local(zero_form, c)
                else:
                    #A = assemble_local(F_solid, c)
                    A = assemble_local(F_solid, c)		# solid = solid - fluid
                if F_fluid_interface_x != None:
                    A_int = integrate_interface(points[1], points[2], c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                    A = np.array(A) + np.array(A_int)
            else:
                cut = 1
                A = vertex_edge_integration(F_fluid_cutcell, F_solid_cutcell, 1, 
                    flags, points, new_point1, c)
                if F_fluid_interface_x != None:
                    A_int = integrate_interface(points[0], new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                    A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(new_point1 - points[2]) < closeness_TOL:
            cut = 1
            A = vertex_edge_integration(F_fluid_cutcell, F_solid_cutcell, 2,
                    flags, points, new_point0, c)
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[2], new_point0, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        else:
            cut = 1
            A = edge_edge_integration(F_fluid_cutcell, F_solid_cutcell, 0, flags, 
                                       points, new_point0, new_point1, c)
            if F_fluid_interface_x != None:
                A_int = integrate_interface(new_point0, new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
    elif vertex == 1:
        if np.linalg.norm(points[1] - new_point0) < closeness_TOL:
            if flags[0] == 0:
                #A = assemble_local(F_fluid, c)
                A = assemble_local(zero_form, c)
            else:
                #A = assemble_local(F_solid, c)
                A = assemble_local(F_solid, c)		# solid = solid - fluid
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[1], new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(points[1] - new_point1) < closeness_TOL:
            if flags[2] == 0:
                #A = assemble_local(F_fluid, c)
                A = assemble_local(zero_form, c)
            else:
                #A = assemble_local(F_solid, c)
                A = assemble_local(F_solid, c)		# solid = solid - fluid
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[1], new_point0, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(new_point0 - points[0]) < closeness_TOL:
            if np.linalg.norm(new_point1 - points[2]) < closeness_TOL:
                if flags[1] == 0:
                    #A = assemble_local(F_fluid, c)
                    A = assemble_local(zero_form, c)
                else:
                    #A = assemble_local(F_solid, c)
                    A = assemble_local(F_solid, c)		# solid = solid - fluid
                if F_fluid_interface_x != None:
                    A_int = integrate_interface(points[0], points[2], c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                    A = np.array(A) + np.array(A_int)
            else:
                cut = 1
                A = vertex_edge_integration(F_fluid_cutcell, F_solid_cutcell, 0,
                    flags, points, new_point1, c)
                if F_fluid_interface_x != None:
                    A_int = integrate_interface(points[0], new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                    A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(new_point1 - points[2]) < closeness_TOL:
            cut = 1
            A = vertex_edge_integration(F_fluid_cutcell, F_solid_cutcell, 2,
                    flags, points, new_point0, c)
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[2], new_point0, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        else:
            cut = 1
            A = edge_edge_integration(F_fluid_cutcell, F_solid_cutcell, 1, flags, 
                                       points, new_point0, new_point1, c)
            if F_fluid_interface_x != None:
                A_int = integrate_interface(new_point0, new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
    else: 
        if np.linalg.norm(points[2] - new_point0) < closeness_TOL:
            if flags[1] == 0:
                #A = assemble_local(F_fluid, c)
                A = assemble_local(zero_form, c)
            else:
                #A = assemble_local(F_solid, c)
                A = assemble_local(F_solid, c)		# solid = solid - fluid
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[2], new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(points[2] - new_point1) < closeness_TOL:
            if flags[0] == 0:
                #A = assemble_local(F_fluid, c)
                A = assemble_local(zero_form, c)
            else:
                #A = assemble_local(F_solid, c)
                A = assemble_local(F_solid, c)		# solid = solid - fluid
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[2], new_point0, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(new_point0 - points[1]) < closeness_TOL:
            if np.linalg.norm(new_point1 - points[0]) < closeness_TOL:
                if flags[2] == 0:
                    #A = assemble_local(F_fluid, c)
                    A = assemble_local(zero_form, c)
                else:
                    #A = assemble_local(F_solid, c)
                    A = assemble_local(F_solid, c)		# solid = solid - fluid
                if F_fluid_interface_x != None:
                    A_int = integrate_interface(points[0], points[1], c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                    A = np.array(A) + np.array(A_int)
            else:
                cut = 1
                A = vertex_edge_integration(F_fluid_cutcell, F_solid_cutcell, 1,
                    flags, points, new_point1, c)
                if F_fluid_interface_x != None:
                    A_int = integrate_interface(points[1], new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                    A = np.array(A) + np.array(A_int)
        elif np.linalg.norm(new_point1 - points[0]) < closeness_TOL:
            cut = 1
            A = vertex_edge_integration(F_fluid_cutcell, F_solid_cutcell, 0,
                    flags, points, new_point0, c)
            if F_fluid_interface_x != None:
                A_int = integrate_interface(points[0], new_point0, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)
        else:
            cut = 1
            A = edge_edge_integration(F_fluid_cutcell, F_solid_cutcell, 2, 
                    flags, points, new_point0, new_point1, c)
            if F_fluid_interface_x != None:
                A_int = integrate_interface(new_point0, new_point1, c, 
                        F_fluid_interface_x, F_solid_interface_x, 
                        F_fluid_interface_y, F_solid_interface_y, ips)
                A = np.array(A) + np.array(A_int)

    if cut == 1:
        #print('cut!')
        if c.orientation() == 1: # 1 counterclockwise, 0 clockwise (probably)
            A *= -1

        #A_fluid = assemble_local(F_fluid, c)
        ##print(A_fluid.size, len(A_fluid))
        #if type(A_fluid) != np.float64 and A_fluid.size == len(A_fluid):
        #    A_fluid = A_fluid.reshape(len(A_fluid), 1)
        #    #A.reshape(len(A_fluid), 1)
        ##print('A_fluid: ', A_fluid.shape)
        ##print(type(A_fluid), type(A))
        #A = A - A_fluid

    #print('divide and integrate: ', A.shape)
    return A

def integrate(F_fluid, F_fluid_cutcell, F_solid, F_solid_cutcell, zero_form, points, flags, ips, c,
        F_fluid_interface_x=None, F_solid_interface_x=None, 
        F_fluid_interface_y=None, F_solid_interface_y=None, ):
    """
    finds how to integrate (whether the cell is intersected or not) and sends to other subroutines.
    """
    #info("integrate")
    #cut = False
    test = 0
    if test:
        A_loc = assemble_local(zero_form, c)
    else:
        if flags[0] == flags[1] == flags[2]:
            if flags[0] == 0:
                #info('assemble local - zero')
                #A_loc = assemble_local(F_fluid, c)
                A_loc = assemble_local(zero_form, c)
            else: 
                #cut = True
                #info('assemble local - solid')
                #A_loc = assemble_local(F_solid, c)
                A_loc = assemble_local(F_solid, c)       # solid = solid - fluid

        else:
            #cut = True
            vertex, new_point0, new_point1 = find_division_points(flags, points, ips)

            A_loc = divide_and_integrate(F_fluid, F_fluid_cutcell, F_solid, F_solid_cutcell, 
                    zero_form, vertex, flags, points, new_point0, new_point1, c, ips,
                    F_fluid_interface_x, F_solid_interface_x, 
                    F_fluid_interface_y, F_solid_interface_y)
    #if cut: info('cut')
    #else: info('notcut')
    #print('integrate: ', A_loc.shape)
    return A_loc#, cut

def assemble_cutcell(A, F_fluid, F_fluid_cutcell, F_solid, F_solid_cutcell, zero_form,
        F_fluid_interface_x, F_solid_interface_x, F_fluid_interface_y, F_solid_interface_y,
        cutcells, V, ips):
    """
    Performes for-cycle over cells stored in 'cutcells', those are the cells which could be 
    intersected by the FSI-interface. Assembles locally each of this cells (uses corresponding
    form, Fluid, Solid, or both in the same time - in this case integrates each form just
    over its part) and stores it in the matrix A.

    For integration over cells uses FEniCS method assemble_local, if the cell is not intersected.
    Otherwise, it founds points of intersection (find_division_points) and then integrates fluid
    and solidform over the cell, each over its corresponding parti (divide_and_integrate).
    The routines for cutcell integration are defined above in this file.
    """
    #info("assemble_cucell")
    E = as_backend_type(A).mat()
    E.setUp()
    #info('len: {}'.format(len(cutcells)))
    for c in cutcells:
        #info('for loop for cells started')
        dofs = V.dofmap().cell_dofs(c.index())
        verts_coord = c.get_vertex_coordinates()

        points = [np.array(verts_coord[2*i:2*i+2]) for i in range(3)]

        flags = [ips(point) for point in points]

        A_loc = integrate(F_fluid, F_fluid_cutcell, F_solid, F_solid_cutcell, zero_form,
                points, flags, ips, c, F_fluid_interface_x, F_solid_interface_x, 
                F_fluid_interface_y, F_solid_interface_y)
 
        val = A_loc.reshape(-1)

        #info('setValues')
        E.setValuesLocal(dofs, dofs, val, addv=True)

    E.assemble()


def assemble_cutcell_time_dependent(A, F_fluid, F_fluid0, F_fluid_cutcell, F_fluid_cutcell0, 
                   F_solid, F_solid0, F_solid_cutcell, F_solid_cutcell0, zero_form,
                   cutcells, V, ips, ips0):
    """
    Performes integration over possibly cut cells. Forms from current time step integrates
    according to the current position of FSI-interface('ips'), to integrate forms from the previous
    time step it uses corresponding interface position ('ips0').

    """
    #info('assemble_cucell_time_dependent')
    #info('ips: {}'.format(ips(0.4,0.2)))
    E = as_backend_type(A).vec()
    E.setUp()

    new_cutcells = set()
    #info('len before loop: {}'.format(len(new_cutcells)))

    #info('before loop over cells')
    for c in cutcells:
        #info('for loop for cells started')
        #info('define dofmap')
        dofs = V.dofmap().cell_dofs(c.index())      # local dof
        #info('extract coordinates')
        verts_coord = c.get_vertex_coordinates()

        points = [np.array(verts_coord[2*i:2*i+2]) for i in range(3)]

        #info('call ips')
        flags = [ips(point) for point in points]
        A_loc = integrate(F_fluid, F_fluid_cutcell, F_solid, F_solid_cutcell, zero_form, 
                points, flags, ips, c)

        if 0: #cut:
            #info('cut')
            new_cutcells.add(c)
            for cc in cells(c):
                new_cutcells.add(cc)

        #info('call ips0')
        flags0 = [ips0(point) for point in points]
        A_loc0 = integrate(F_fluid0, F_fluid_cutcell0, F_solid0, F_solid_cutcell0, zero_form,
                points, flags0, ips0, c)

        #info('sum and reshape')
        A_loc = np.array(A_loc.reshape(-1)) + np.array(A_loc0.reshape(-1))
        val = A_loc.reshape(-1)

        #info('setValues')
        E.setValuesLocal(dofs, val, addv=True)
    #info('assemble')
    E.assemble()

    #info('len: {}'.format(len(new_cutcells)))
    return new_cutcells
