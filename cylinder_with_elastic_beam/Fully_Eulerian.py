#######################################################################
###
###		Integration over intersected elements according to
###		>>J. Hoffman, B. Holm, T. Richter, The locally adapted patch finite element method for
###		interface problems on triangular meshes, in Fluid-Structure Interaction. 
###             Modeling, Adaptive Discretisations and Solvers. 
###             Radon Series on Computational and Applied Mathematics, vol. 20
###		(de Gruyter, Berlin, 2017)<<
###
###		Initial Point Set function gives 0 if in fluid, 1 if in solid
###		    - looks like it works good
###
###             move integrating routines inside NonlinearProblem (maybe define it in another file)
###
###		Rewrite time derivative (F_temp) to use correct values for functions from 
###             previous time step
###
#######################################################################

from dolfin import *
import mshr
import numpy as np
import sys
import csv

# Use UFLACS to speed-up assembly and limit quadrature degree
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4


class Flow(object):
    def __init__(self, mesh, bndry, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result, 
            *args, **kwargs):

        result = result

        self.mesh  = mesh
        self.bndry = bndry
        self.dt    = dt
        self.t     = 0.0
        self.v_max = v_max

        self.mu_f     = mu_f
        self.rho_f    = rho_f
        self.lambda_s = lambda_s
        self.mu_s     = mu_s
        self.rho_s    = rho_s

        # Define finite elements
        eV = VectorElement("CG", mesh.ufl_cell(), 2)
        eU = VectorElement("CG", mesh.ufl_cell(), 2)
        eP = FiniteElement("CG", mesh.ufl_cell(), 1)
        eR = FiniteElement("R", mesh.ufl_cell(), 0)
        eW = MixedElement([eV, eU, eP, eR])
        W  = FunctionSpace(self.mesh, eW)
        self.W = W
    
        # Set boundary conditions
        self.v_in = Expression(("t<2.0? 0.5*(1.0 - cos(0.5*pi*t))*v_max*4/(gW*gW)*(x[1]*(gW - x[1])):\
                v_max*4/(gW*gW)*(x[1]*(gW - x[1]))", "0.0"),\
                degree = 2, v_max = Constant(self.v_max), gW = Constant(gW), t = self.t)

        bc_u_in     = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _INFLOW)
        bc_u_walls  = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_u_out    = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _OUTFLOW)
        bc_u_circle = DirichletBC(self.W.sub(1), Constant((0.0, 0.0)), bndry, _CIRCLE)
        bc_v_in     = DirichletBC(self.W.sub(0), self.v_in,            bndry, _INFLOW)
        bc_v_walls  = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _WALLS)
        bc_v_circle = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), bndry, _CIRCLE)

        self.bcs = [bc_v_in, bc_v_walls, bc_v_circle, bc_u_in, bc_u_walls, bc_u_circle, bc_u_circle]

        self.n = FacetNormal(self.mesh)
        I = Identity(self.W.mesh().geometry().dim())

        # Define test functions
        (v_, u_, p_, c_) = TestFunctions(self.W)

        # current unknown time step
        self.w = Function(self.W)
        (self.v, self.u, self.p, self.c) = split(self.w)
        #(self.v, self.u, self.p, self.c, self.d) = split(self.w)

        # previous known time step
        self.w0 = Function(self.W)
        (self.v0, self.u0, self.p0, self.c0) = split(self.w0)
        #(self.v0, self.u0, self.p0, self.c0, self.d0) = split(self.w0)

        
        # interface tracking method
        class IPS(Expression):
            """
            Initial Point Set function
            returns 0 if x belongs to fluid domain,
                    1 if x belongs to solid domain,
            """
            def __init__(self, u, **kwargs):
                self.u = u

            def eval(self, value, x):
                if type(x) == Point:
                    a = x.x()
                    b = x.y() 
                elif type(x) == np.ndarray or type(x) == list:
                    a = x[0]
                    b = x[1]
                else:
                    raise NameError('Invalid_type_of_x')
                    
                IPS0 = a - self.u[0]((a, b))
                IPS1 = b - self.u[1]((a, b))
                value[0] = 1 if (IPS0>gLs1 and IPS0<gLs2 and IPS1>gWs1 and IPS1<gWs2) else 0
                #value[0] = 0		# fluid flowing around rigid circle
            def value_shape(self):
                return ()

        ips  = IPS(self.u , degree = 0)		# Initial Point Set function in current time step
        ips0 = IPS(self.u0, degree = 0)		# Initial Point Set function in previous time step
                     # How to use this? (assuming I can integrate)
                     #     * (ips==n)*inner(u,v)*dx			- condition evaluated as False everytime
                     #     * (inner(u,v) if ips==n else 0.0)*dx		- evaluates 'inner(u, v)' only when condition satisfied 
                     #								(which is never, according to previous point)
                     #     * a(ips-b)(ips-c)*inner(u, v)*dx		- stupid, but works (slow iterations)
                     #		
                     # 		- need to find a way, how to evaluste ips(x)
                     #			* write F_fluid like an expression


        # stupid way - (ips==n) in equations gives False everytime
        is_fluid  = 0.5*(ips  - 1)*(ips  - 2)
        is_fluid0 = 0.5*(ips0 - 1)*(ips0 - 2)

        is_solid  = - ips *(ips  - 2)
        is_solid0 = - ips0*(ips0 - 2)

        is_interface  = 0.5*(ips  - 1)*ips
        is_interface0 = 0.5*(ips0 - 1)*ips0



#        def integrate_new_cell(triangle, F_T, F):
#            '''
#            integrates function F over triangle with vertices x0, x1 and x2
#            for this uses some quadrature rule
#            this function is used in 'edge_edge_integration' and 'vertex_edge_integration'
#            which itegrates over cut reference element
#
#            see (J. Hoffman, B. Holm, T. Richter) page 6
#
#            Input - triangle (0, 1, 2, 3) - on which subtriangle of reference triangle we integrate
#                  - F_T                   - maping from reference domain to element on interface
#                  - F                     - function to integrate
#            '''
#            if triangle == 0:
#                return(

        def give_quadrature_points_and_weights(degree):
            '''
            returns quadrature points and weights for integration
            on reference triangle, according to
            'FIAT/quadrature_schemes.py'
            '''
            if degree == 0 or degree == 1:
                # Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
                x = array([[1.0/3.0, 1.0/3.0]])
                w = array([0.5])
            elif degree == 2:
                # Scheme from Strang and Fix, 3 points, degree of precision 2
                x = np.array([[1.0/6.0, 1.0/6.0],
                           [1.0/6.0, 2.0/3.0],
                           [2.0/3.0, 1.0/6.0]])
                w = np.arange(3, dtype=np.float64)
                w[:] = 1.0/6.0
            elif degree == 3:
                # Scheme from Strang and Fix, 6 points, degree of precision 3
                x = np.array([[0.659027622374092, 0.231933368553031],
                           [0.659027622374092, 0.109039009072877],
                           [0.231933368553031, 0.659027622374092],
                           [0.231933368553031, 0.109039009072877],
                           [0.109039009072877, 0.659027622374092],
                           [0.109039009072877, 0.231933368553031]])
                w = np.arange(6, dtype=np.float64)
                w[:] = 1.0/12.0
            elif degree == 4:
                # Scheme from Strang and Fix, 6 points, degree of precision 4
                x = np.array([[0.816847572980459, 0.091576213509771],
                           [0.091576213509771, 0.816847572980459],
                           [0.091576213509771, 0.091576213509771],
                   [0.108103018168070, 0.445948490915965],
                   [0.445948490915965, 0.108103018168070],
                   [0.445948490915965, 0.445948490915965]])
                w = np.arange(6, dtype=np.float64)
                w[0:3] = 0.109951743655322
                w[3:6] = 0.223381589678011
                w = w/2.0
            return(x, w/4.0)


        def edge_edge_integration(q, r, s, p, i, F_T, F_fluid, F_solid):
            '''
            performs integration on given triangle, 
            divide it into 4 smaller, where each is treated purely as solid or fluid element,
            and then calls integrate_new_cell() which integrates over each of the 4 new triangles

            Input - q, r, s          - intersection coefficients
                  - p                - array of three numbers {0, 1} indicating 
                                          whether vertices are in fluid or solid domain
                  - i                - number of edge which is not intersected 
                                              (edge opposite to vertex i)
                  - F_T              - mapping from reference element to the real one
                  - F_solid, F_fluid - varitaional formulations

            Output - value of the integral over the reference element specified by given input
            '''
            degree = 2
            Q, w = give_quadrature_points_and_weights(degree)
            def map_to_element(P, i):
                '''
                maps point p from reference triangle to one of 4 subtriangles
                which originate from intersection by interface
                i in {0, 1, 2} indicates which subtriangle

                Output: Tuple of mapped point coordinates
                '''
                if type(P) == Point:
                    x = P.x()
                    y = P.y()
                elif type(P) == np.ndarray or type(P) == list:
                    x = P[0]
                    y = P[1]
                else:
                    raise NameError('Invalid_type_of_p')
                if i == 0:
                    return(F_T((s*x, q*y)))
                elif i == 1:
                    return(F_T((s + (1.0 - s)*x + (1.0 - s - r)*y, r*y)))
                elif i == 2:
                    return(F_T(((1.0 - r)*x, q + (r - q)*q + (1.0 - q)*y)))
                elif i == 3:
                    return(F_T((1.0 - r + (r - 1.0)*x + (r + s - 1.0)*y, \
                            r + (q - r)*x - r*y)))
                else:
                    raise NameError('Invalid_value_of_i')
            
            I = 0.0

            for j in range(3):
                F = F_fluid if p[i] == 0 else F_solid
                I += sum([w[i]*F(map_to_element(Q[i], j)) for i in range(len(Q))])
            if sum(np.equal(0, p)) == 2:
                I += sum([w[i]*F_fluid(map_to_element(Q[i], 3)) for i in range(len(Q))])
            else:
                I += sum([w[i]*F_solid(map_to_element(Q[i], 3)) for i in range(len(Q))])
                
            return(I)


        def vertex_edge_integration(q, r, s, p, vertex, F_T, F_fluid, F_solid):
            '''
            performs integration on given triangle, 
            divide it into 4 smaller, where each is treated as either solid either fluid element,
            and then calls integrate_new_cell() which integrates over each of the 4 new triangles

            Input - q, r, s          - intersection coefficients
                  - p                - array of three numbers {0, 1} indicating 
                                          whether vertices are in fluid or solid domain
                  - vertex           - number of vertex which is intersected
                  - F_T              - mapping from reference element to the real one
                  - F_solid, F_fluid - varitaional formulations

            Output - value of the integral over the reference element specified by given input
            '''
            degree = 2
            Q, w = give_quadrature_points_and_weights(degree)
            def map_to_element(P, triangle, vertex):
                '''
                maps point p from reference triangle to one of 4 subtriangles
                which originate from intersection by interface
                triangle in {0, 1, 2} indicates which subtriangle
                vertex is the index of intersected vertex

                Output: Tuple of mapped point coordinates
                '''
                if type(P) == Point:
                    x = P.x()
                    y = P.y()
                elif type(P) == np.ndarray or type(P) == list:
                    x = P[0]
                    y = P[1]
                else:
                    raise NameError('Invalid_type_of_P')
                if vertex == 0:
                    if triangle == 0:
                        return(F_T(((1.0-r)*y, q - q*x + (r - q)*y)))
                    elif triangle == 1:
                        return(F_T((s + (1.0 - r -s)*x -s*y, r*x)))
                    elif triangle == 2:
                        return(F_T((s + (1.0 - s)*x + (1.0 - s - r)*y, r*y)))
                    elif triangle == 3:
                        return(F_T(((1.0 - q)*x, q + (r - q)*x + (1.0 - q)*y)))
                    else:
                        raise NameError('Invalid_value_of_triangle')
                elif vertex == 1:
                    if triangle == 0:
                        return(F_T((s*x, q*y)))
                    elif triangle == 1:
                        return(F_T((s + (1.0 - s)*x, q*y)))
                    elif triangle == 2:
                        return(F_T((1.0 - r + (r - 1.0)*x + r*y,\
                            r + (q - r)*x - r*y)))
                    elif triangle == 3:
                        return(F_T(((1.0 - r)*x, q + (r-q)*x + (1.0 - q)*y)))
                    else:
                        raise NameError('Invalid_value_of_triangle')
                elif vertex == 2:
                    if triangle == 0:
                        return(F_T((s*x, q*y)))
                    elif triangle == 1:
                        return(F_T((s*x, q - q*x + (1.0 - q)*y)))
                    elif triangle == 2:
                        return(F_T((1.0 - r + (r - 1.0)*x + (r + s - 1.0)*y,\
                            r + (1.0 - r)*x - r*y)))
                    else:
                        raise NameError('Invalid_value_of_triangle')
                else:
                    raise NameError('Invalid_value_of_vertex')

            I = 0

            if vertex == 0:
                F1, F2 = (F_fluid, F_solid) if p[1] == 0 else (F_solid, F_fluid)
                I += sum([w[i]*F1(map_to_element(Q[i], 0, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F1(map_to_element(Q[i], 3, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F2(map_to_element(Q[i], 1, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F2(map_to_element(Q[i], 2, vertex)) for i in range(len(Q))])

            elif vertex == 1:
                F1, F2 = (F_fluid, F_solid) if p[0] == 0 else (F_solid, F_fluid)
                I += sum([w[i]*F1(map_to_element(Q[i], 0, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F1(map_to_element(Q[i], 1, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F2(map_to_element(Q[i], 2, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F2(map_to_element(Q[i], 3, vertex)) for i in range(len(Q))])

            elif vertex == 0:
                F1, F2 = (F_fluid, F_solid) if p[0] == 0 else (F_solid, F_fluid)
                I += sum([w[i]*F1(map_to_element(Q[i], 0, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F1(map_to_element(Q[i], 1, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F2(map_to_element(Q[i], 2, vertex)) for i in range(len(Q))])
                I += sum([w[i]*F2(map_to_element(Q[i], 3, vertex)) for i in range(len(Q))])
            return(I)



        def integrate_non_intersected_element(c, F):
            '''
            performs integration of function F over given cell c
            '''
            return assemble_local(F*dx(1), c)
            


        def find_point(x, y, px, ips):
            '''
            finds point where the interface intersects given edge of element
            '''
            TOL = 1.0e-12      # ????	
            #TOL2 = 1.0e-12      # ????		# squared tolerance
            #m = np.multiply(0.5, (np.add(x, y)))
            m = 0.5*(x + y)
            if x.distance(y) < TOL:
            #if ( (x[0] - y[0])**2 + (x[1] - y[1])**2 ) < TOL2:
                return m 
            else:
                #if ips((m.x(), m.y())) == px:
                if ips(m) == px:
                #if ips([m[0], m[1]]) == px:
                    return(find_point(m, y, px, ips))
                else:
                    return(find_point(x, m, px, ips))



        def cut_integrate(c, p0, p1, p2, ips, F_fluid, F_solid):
            '''
            Computes integrals over element c, which is cut by interface

            between values p0, p1, p2 are two with the same value and one with different, 
            that means, that the interface goes through this element. This subroutine 
            at first finds the affine mapping between this cell and reference element.
            Then finds the points of intersection and decide how to integrate.
            For integration it calls corresponding functions.

            Input - c   - cut cell
                    p0  - indicates position of vertex x[0] of the cell c (0 if in fluid domain and 1 if in solid domain)
                    p1  - indicates position of vertex x[1] of the cell c (0 if in fluid domain and 1 if in solid domain)
                    p2  - indicates position of vertex x[2] of the cell c (0 if in fluid domain and 1 if in solid domain)
                    ips - Initial Point Set Function
                    F_fluid - variational formulation for fluid problem
                    F_solid - variational formulation for solid problem

            Output - value of the integral
            '''
            # extract vertices of the cell
            x = c.get_vertex_coordinates()
            x0, x1, x2 = Point(x[0], x[1]), Point(x[2], x[3]), Point(x[4], x[5])
            #x0, x1, x2 = [x[0], x[1]], [x[2], x[3]], [x[4], x[5]]

            #compute matrix of affine mapping (A), its determinant (det_A) and inverse (inv_A)
            A = np.array([[ x1.x() - x0.x(), x2.x() - x0.x() ], \
                          [ x1.y() - x0.y(), x2.y() - x0.y() ]])
            #A = np.array([[ x1[0] - x0[0], x2[0] - x0[0] ], \
            #              [ x1[1] - x0[1], x2[1] - x0[1] ]])
            det_A = A[0][0]*A[1][1] - A[0][1]*A[1][0]
            inv_A = (1.0/det_A)*np.array([[  A[1][1], -A[0][1] ], \
                                          [ -A[1][0],  A[0][0] ] ])
            # affine mapping
            def F_T(y):
                return(np.dot(A[0], y) + x0.x(), np.dot(A[1], y) + x0.y())

            I = 0

            # set tolerance for how close needs two point need to be to treat like one point
            closeness_TOL = 1e-8                 #??????

            # perform the integration - find intersection points and then integrate over reference triangle
            if p0 == p1:	# exactly two vertices are in the same domain
                y = find_point(x0, x2, p0, ips)
                z = find_point(x1, x2, p1, ips)
                if y.distance(x0) < closeness_TOL:	
                    if z.distance(x1) < closeness_TOL:	# interface goes almost the same as the edge - integrate as non-intersected element
                        if p2 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    elif z.distance(x2) < closeness_TOL:
                        if p1 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    else:		# interface goes through the vertex and edge
                        P = [a[0]*(z-x0).x() + a[1]*(z-x0).y() for a in inv_A]
                        r = 1.0 - P[0]
                        if r < 0.5:
                            s = 1 - r
                            q = 0.5
                        else:
                            s = 0.5
                            q = r
                        I = vertex_edge_integration(q, r, s, [p0, p1, p2], 0, F_T, F_fluid, F_solid)*det_A

                elif y.distance(x2) < closeness_TOL:	
                    if z.distance(x1) < closeness_TOL:	# interface goes almost the same as the edge - integrate as non-intersected element
                        if p0 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    elif z.distance(x2) < closeness_TOL:
                        if p0 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    else:		# interface goes through the vertex and edge
                        # should not happen
                        raise NameError('Strange_triangle')
                else: 			# interface goes through two edges
                    P = [a[0]*(y-x0).x() + a[1]*(y-x0).y() for a in inv_A]
                    q = P[1]
                    if z.distance(x1) < closeness_TOL:
                        if q < 0.5:
                            s = q
                            r = 0.5
                        else:
                            s = 0.5
                            r = q
                        I = vertex_edge_integration(q, r, s, [p0, p1, p2], 1, F_T, F_fluid, F_solid)*det_A
                    elif z.distance(x2) < closeness_TOL:
                        if p1 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    else:
                        Q = [a[0]*(z-x0).x() + a[1]*(z-x0).y() for a in inv_A]
                        r = 1.0 - Q[0]
                        s = q if (q<0.5 and r<0.5) else 0.5
                        I = edge_edge_integration(q, r, s, [p0, p1, p2], 1, F_T, F_fluid, F_solid)*det_A
            elif p0 == p2:
                y = find_point(x0, x1, p0, ips)
                z = find_point(x1, x2, p1, ips)
                if y.distance(x0) < closeness_TOL:	
                    if z.distance(x1) < closeness_TOL:	# interface goes almost the same as the edge - integrate as non-intersected element
                        if p2 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    elif z.distance(x2) < closeness_TOL:
                        if p1 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    else:
                        P = [a[0]*(z-x0).x() + a[1]*(z-x0).y() for a in inv_A]	# find point on the reference element edge
                        r = 1.0 - P[0]
                        if r < 0.5:
                            s = 1 - r
                            q = 0.5
                        else:
                            s = 0.5
                            q = r
                        I = vertex_edge_integration(q, r, s, [p0, p1, p2], 0, F_T, F_fluid, F_solid)*det_A
                elif y.distance(x1) < closeness_TOL:
                    if z.distance(x1) < closeness_TOL:	# interface goes almost the same as the edge - integrate as non-intersected element
                        if p2 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    elif z.distance(x2) < closeness_TOL:
                        if p0 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    else:
                        # should not happen
                        raise NameError('Strange_triangle')
                else: 			# interface goes through two edges
                    P = [a[0]*(y-x0).x() + a[1]*(y-x0).y() for a in inv_A]
                    s = P[0]
                    if z.distance(x1) < closeness_TOL:	# interface goes almost the same as the edge - integrate as non-intersected element
                        if p2 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    elif z.distance(x2) < closeness_TOL:
                        if s < 0.5:
                            r = 0.5
                            q = s
                        else:
                            r = 1.0 - s
                            q = 0.5
                        I = vertex_edge_integration(q, r, s, [p0, p1, p2], 1, F_T, F_fluid, F_solid)*det_A
                    else:		# interface goes through the vertex and edge
                        Q = [a[0]*(z-x0).x() + a[1]*(z-x0).y() for a in inv_A]
                        q = P[1]
                        r = 1.0 - Q[0]
                        q = s if(r>0.5 and s<0.5) else 0.5
                        I = edge_edge_integration(q, r, s, [p0, p1, p2], 2, F_T, F_fluid, F_solid)*det_A
            elif p1 == p2:
                y = find_point(x0, x2, p0, ips)
                z = find_point(x0, x1, p0, ips)
                if y.distance(x0) < closeness_TOL:
                    if p2 == 0:
                        I = integrate_non_intersected_element(c, F_fluid)
                    else:
                        I = integrate_non_intersected_element(c, F_solid)
                elif y.distance(x2) < closeness_TOL:
                    if z.distance(x0) < closeness_TOL:
                        if p1 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    elif z.distance(x1) < closeness_TOL:
                        if p0 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                else:
                    P = [a[0]*(y-x0).x() + a[1]*(y-x0).y() for a in inv_A]
                    q = P[1]
                    if z.distance(x0) < closeness_TOL:
                        if p1 == 0:
                            I = integrate_non_intersected_element(c, F_fluid)  
                        else:
                            I = integrate_non_intersected_element(c, F_solid)
                    elif z.distance(x1) < closeness_TOL:
                        if q < 0.5:
                            s = q
                            r = 0.5
                        else:
                            s = 0.5
                            r = q
                        I = vertex_edge_integration(q, r, s, [p0, p1, p2], 1, F_T, F_fluid, F_solid)*det_A
                    else:
                        Q = [a[0]*(z-x0).x() + a[1]*(z-x0).y() for a in inv_A]
                        s = Q[0]
                        r = q if (q>0.5 and s>0.5) else 0.5
                        I = edge_edge_integration(q, r, s, [p0, p1, p2], 0, F_T, F_fluid, F_solid)*det_A
            else:
                raise NameError('Invalid_triangle')
            return 0

        def integrate_over_interface(u, F_fluid, F_solid): 
            '''
            Goes through all cells which can be intersected and computes integrals
            for variational form.

            For doing this it uses reference element E has coordinates e0 = (0,0), e1 = (1,0) and e2=(0,1)
            element T with vertices x0, x1, x2 is an affine transformation of E: x_i = x0 + A*e_i,   i=1,2

            Input: - dissplacement u - can be used for self.u or self.u0
                   - F_fluid - variational form of fluid problem (without *dx)
                   - F_solid - variational form of solid problem (without *dx)

            Output - value of the integral
            '''
            I = 0
            ips = IPS(u, degree=0)
            for c in cells(mesh):		# is it possible to iterate only over cells, with value regions[1]???
                if regions[c] == 1:
                    x = c.get_vertex_coordinates()
                    p0, p1, p2 = ips(x[0:2]), ips(x[2:4]), ips(x[4:])
                    if p0 == p1 == p2 == 0:
                        I += integrate_non_intersected_element(c, F_fluid)
                    elif p0 == p1 == p2 == 1:
                        I += integrate_non_intersected_element(c, F_solid)
                    else:
                        I += cut_integrate(c, p0, p1, p2, ips, F_fluid, F_solid)
            return I
 


        # define deformation gradient and Jacobian
        self.FF  = I + grad(self.u)
        self.FF0 = I + grad(self.u0)
        JJ  = det(self.FF)
        JJ0 = det(self.FF0)

        # write equations for fluid
        D  = sym(grad(self.v))
        D0 = sym(grad(self.v0))
        a_fluid =  inner(2*self.mu_f*D, grad(v_)) + inner(self.rho_f*grad(self.v)*self.v, v_)
        a_fluid0 = inner(2*self.mu_f*D0, grad(v_)) + inner(self.rho_f*grad(self.v0)*self.v0, v_)
       
        b_fluid = inner(self.p, div(v_)) + inner(p_, div(self.v))
        b_fluid0 = inner(self.p0, div(v_)) + inner(p_, div(self.v0))
        
        #F_fluid = (theta*((a_fluid + b_fluid + inner(grad(self.u), grad(u_))) if ips == 0 else 0.0)\
        #       + (1.0 - theta)*((a_fluid0 + b_fluid0 + inner(grad(self.u0), grad(u_))) if ips0 == 0 else 0.0))*dx
        #F_fluid = theta*(a_fluid + b_fluid + inner(grad(self.u), grad(u_)))*(ips == 0)*dx \
        #       + (1.0 - theta)*(a_fluid0 + b_fluid0 + inner(grad(self.u0), grad(u_)))*(ips0 == 0)*dx
        #F_fluid = theta*is_fluid*(a_fluid + b_fluid + inner(grad(self.u), grad(u_)))*dx \
        #       + (1.0 - theta)*is_fluid0*(a_fluid0 + b_fluid0 + inner(grad(self.u0), grad(u_)))*dx
        #F_fluid = (theta*(a_fluid + b_fluid + inner(grad(self.u), grad(u_))) \
        #       + (1.0 - theta)*(a_fluid0 + b_fluid0 + inner(grad(self.u0), grad(u_))))*dx(0) \
        #     + theta*is_fluid*(a_fluid + b_fluid + inner(grad(self.u), grad(u_)))*dx(1) \
        #       + (1.0 - theta)*is_fluid0*(a_fluid0 + b_fluid0 + inner(grad(self.u0), grad(u_)))*dx(1)

    
        # write equations for solid
        E  = 0.5*(inv(self.FF.T )*inv(self.FF ) - I)
        E0 = 0.5*(inv(self.FF0.T)*inv(self.FF0) - I)
        T_s  = JJ*inv(self.FF  )*(2*self.mu_s*E  + self.lambda_s*tr(E )*I)*inv(self.FF.T) 		# Cauchy stress 
        T_s0 = JJ0*inv(self.FF0)*(2*self.mu_s*E0 + self.lambda_s*tr(E0)*I)*inv(self.FF0.T)		# Cauchy stress 

        a_solid  = inner((self.rho_s/JJ )*grad(self.v )*self.v,  v_)
        a_solid0 = inner((self.rho_s/JJ0)*grad(self.v0)*self.v0, v_)

        b_solid  = inner(grad(self.u )*self.v  - self.v , u_)
        b_solid0 = inner(grad(self.u0)*self.v0 - self.v0, u_)

        #F_solid = (theta*((inner(T_s, grad(v_)) + a_solid + b_solid) if ips == 1 else 0.0) \
        #          + (1.0 - theta)*((inner(T_s0, grad(v_)) + a_solid0 + b_solid0) if ips0 ==1 else 0.0))*dx
        #F_solid = theta*is_solid*(inner(T_s, grad(v_)) + a_solid + b_solid)*dx(1) \
        #          + (1.0 - theta)*is_solid0*(inner(T_s0, grad(v_)) + a_solid0 + b_solid0)*dx(1)

        # contraint on zero mean pressure on outwlow
        F_press = theta*(self.p*c_*ds(_OUTFLOW) + p_*self.c*ds(_OUTFLOW)) + (1.0 - theta)*(self.p0*c_*ds(_OUTFLOW) + p_*self.c0*ds(_OUTFLOW))

        # temporal derivative terms
        #F_temp = (self.rho_f*(1/self.dt)*inner(self.v - self.v0, v_) if ips == 0 else 0.0)*dx \
        #       + ((self.rho_s/JJ)*(1/self.dt)*inner(self.v - self.v0, v_) if ips == 1 else 0.0)*dx
        #F_temp = self.rho_f*(1/self.dt)*inner(self.v - self.v0, v_)*dx(0) \
        #       + self.rho_f*(1/self.dt)*inner(self.v - self.v0, v_)*is_fluid*dx(1) \
        #       + (self.rho_s/JJ)*(1/self.dt)*inner(self.v - self.v0, v_)*is_solid*dx(1)


        #F = F_temp + F_fluid + F_solid + F_press
        #F = F_temp + F_fluid + F_press
        
        F_fluid  = a_fluid  + b_fluid  + inner(grad(self.u ), grad(u_))
        F_fluid0 = a_fluid0 + b_fluid0 + inner(grad(self.u0), grad(u_))
        F_solid  = inner(T_s , grad(v_)) + a_solid  + b_solid
        F_solid0 = inner(T_s0, grad(v_)) + a_solid0 + b_solid0
        F_temp = self.rho_f*(1/self.dt)*inner(self.v - self.v0, v_)*dx(0) \
               + integrate_over_interface(self.u, self.rho_f*(1/self.dt)*inner(self.v - self.v0, v_)*is_fluid, \
               + (self.rho_s/JJ)*(1/self.dt)*inner(self.v - self.v0, v_) )
        F = theta*(F_fluid*dx(0) + integrate_over_interface(self.u, F_fluid, F_solid) )\
           + (1.0 - theta)*(F_fluid0*dx(0) + integrate_over_interface(self.u0, F_fluid0, F_solid0)) \
           + F_press + F_temp


        J = derivative(F, self.w)

        # configure solver parameters
        self.problem = NonlinearVariationalProblem(F, self.w, bcs=self.bcs, J=J)
        self.solver = NonlinearVariationalSolver(self.problem) 
        self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6 
        self.solver.parameters['newton_solver']['linear_solver'] = 'mumps' 
 #       self.solver.parameters['newton_solver']['error_on_nonconvergence'] = False 
        self.solver.parameters['newton_solver']['maximum_iterations'] = 15 
        PETScOptions.set('mat_mumps_icntl_24', 1)			# detects null pivots
        PETScOptions.set('mat_mumps_cntl_1', 0.01)			# set treshold for partial treshold pivoting, 0.01 is default value
  
        # create files for saving 
        self.vfile = XDMFFile("%s/velo.xdmf" % result) 
        self.ufile = XDMFFile("%s/disp.xdmf" % result) 
        self.pfile = XDMFFile("%s/pres.xdmf" % result)
        self.sfile = XDMFFile("%s/stress.xdmf" % result) 
        self.vfile.parameters["flush_output"] = True 
        self.ufile.parameters["flush_output"] = True 
        self.pfile.parameters["flush_output"] = True 
        self.sfile.parameters["flush_output"] = True
       

    def solve(self, t, dt): 
        self.t = t 
        self.dt = dt 
        self.v_in.t = t
        self.solver.solve()
        self.w0.assign(self.w)
  #      status_solved = self.solver.solve()               # err on nonconvergence disabled         
  #      solved = status_solved[1]                         # use this control instead 
  #      if solved:
  #          (self.u, self.v, self.p) = self.w.split(True) 
  #          self.w0.assign(self.w) 
  #      return solved 

    def save(self, t):
        (v, u, p, c) = self.w.split() 

        v.rename("v", "velocity") 
        u.rename("u", "displacement") 
        p.rename("p", "pressure") 
        self.vfile.write(v, t) 
        self.ufile.write(u, t)
        self.pfile.write(p, t) 
        #s.rename("s", "stress") 
        #self.sfile.write(s, t)


result = "results_Fully_Eulerian"		# name of folder containing results
# load mesh with boundary and domain markers
import marker
(mesh, bndry, interface, unelastic_surface, domains, A, B) \
        = marker.give_marked_mesh(mesh_coarseness = 50, refine = True, ALE = False)

# domain (used while building mesh) - needed for inflow condition(gW) 
#       and Initial Point Set function (rest)
gW   = 0.41			# width of domain
gLs1 = 0.2			# x-coordinate of centre of rigid circle
gLs2 = 0.6 + DOLFIN_EPS		# x-coordinate of end of elastic beam (in initial state)
gWs1 = 0.19 - DOLFIN_EPS	# y-coordinate of beginning of elastic beam
gWs2 = 0.21 + DOLFIN_EPS	# y-coordinate of end of elastic beam

# boundary marks' names (already setted to the mesh) - needed for boundary conditions
_INFLOW  = 1
_WALLS   = 2
_CIRCLE  = 3
_OUTFLOW = 4

# devide domain into 2 regions, one (regions=0) where is surely part of the fluid domain and 
#       second (regions=1), where could elastic solid occur
regions = MeshFunction('size_t', mesh, 2, mesh.domains())
regions.set_all(0)
h = 2*mesh.hmax()	# to let some space after the end of the elastic beam
for f in cells(mesh):
    mp = f.midpoint()
    if mp[0] > 0.2 and mp[0] < gLs2 + h:
        regions[f] = 1

dx  = dx(domain=mesh, subdomain_data = regions)
ds  = ds(domain=mesh, subdomain_data = bndry)

# don't use them yet, will need to rewrite probably
dS  = dS(domain=mesh, subdomain_data = interface)
dss = ds(domain=mesh, subdomain_data = unelastic_surface)

# time stepping
dt    = 0.005
t_end = 10.0
theta = 0.5
v_max = 1.5

rho_s = Constant(10000)
nu_s  = Constant(0.4)
mu_s  = Constant(5e05)
E_s   = Constant(1.4e04)
lambda_s = (mu_s*E_s)/((1+mu_s)*(1-2*mu_s))

rho_f = Constant(1.0e03)
nu_f  = Constant(1.0e-03)
mu_f  = nu_f*rho_f

flow = Flow(mesh, bndry, dt, theta, v_max, lambda_s, mu_s, rho_s, mu_f, rho_f, result)

t = 0.0
while  t < t_end:
    info("t = %.3f, t_end = %.1f" % (t, t_end))
    flow.solve(t, dt)
    #end()
    flow.save(t)

    t += float(dt)

#flow.data.close()
#sys.path.append('../')
#import plotter
#result = "results"		# name of folder containing results
#plotter.plot_all_lag(result+'/data.csv', result+'/mean_press.png', result+'/pressure_jump.png', result+'/A_position.png', \
#         result+'/pressure_difference.png', result+'/drag.png', result+'/lift.png')
