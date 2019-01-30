from dolfin import *
from dolfin import __version__
import mshr

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

def generate_mesh(par, refine, ALE):

    # construct mesh
    geometry = mshr.Rectangle(Point(0.0, 0.0), Point(gL, gW)) - mshr.Circle(Point(gX, gY), g_radius, 20)
    if ALE:
        geometry.set_subdomain(1, mshr.Rectangle(Point(gX, gY - 0.5*gEH), 
            Point(gX + g_radius + gEL, gY + 0.5*gEH)))
    mesh = mshr.generate_mesh(geometry, par)

    if refine:
        parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        for k in range(2):		# 2
            cf = MeshFunction('bool', mesh, 2)
            cf.set_all(False)
            z = 0.08
            for c in cells(mesh):
                if c.midpoint().distance(Point(gX, gY)) < (0.05 + k*z) : cf[c]=True
                elif (c.midpoint()[0] <= gX + g_radius + gEL + k*z and c.midpoint()[0] >= gX - k*z\
                 and c.midpoint()[1] <= gY + 0.5*gEH + k*z and c.midpoint()[1] >= gY - 0.5*gEH - k*z):
                    cf[c] = True
            mesh = refine(mesh, cf)

    mesh.init()

    return(mesh)

def give_marked_mesh(mesh_coarseness = 40, refine = False, ALE = True):
    '''
    Generates mesh and defines boundary and domain classification functions.
    If ALE == True, then the mesh fits the initial position of elastic beam,
    otherwise it is ignored.
    '''

    info("Generating mesh...")

    mesh = generate_mesh(mesh_coarseness, refine, ALE)

    class Cylinder(SubDomain):
        def snap(self, x):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            if r <= g_radius:
                x[0] = gX + (g_radius/r) * (x[0] - gX)
                x[1] = gY + (g_radius/r) * (x[1] - gY)
        def inside(self, x, on_boundary):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            return( (r <= g_radius + DOLFIN_EPS) and on_boundary)

    cylinder = Cylinder()

    class Elasticity(SubDomain):
        def inside(self, x, on_boundary):
            return(x[0] <= gX + g_radius + gEL + DOLFIN_EPS and x[0] >= gX - DOLFIN_EPS\
             and x[1] <= gY + 0.5*gEH + DOLFIN_EPS and x[1] >= gY - 0.5*gEH - DOLFIN_EPS)

    elasticity = Elasticity()

    # construct facet and domain markers
    bndry             = MeshFunction('size_t', mesh, 1)		# boundary conditions marker 
    interface         = MeshFunction('size_t', mesh, 1)		# interface marker
    unelastic_surface = MeshFunction('size_t', mesh, 1)		# circle surface neighbouring with fluid (not with elastic solid)
    domains           = MeshFunction('size_t', mesh, 2, mesh.domains())
    bndry.set_all(0)
    interface.set_all(0)
    unelastic_surface.set_all(0)
    domains.set_all(0)
    elasticity.mark(domains, 1)

    cylinder.mark(bndry, _CIRCLE)
    for f in facets(mesh):
        if f.exterior():
            mp = f.midpoint()
            if near(mp[0], 0.0):
                bndry[f] = _INFLOW 				
            elif near(mp[1], 0.0) or near(mp[1], gW):
                bndry[f] = _WALLS			
            elif near(mp[0], gL):
                bndry[f] = _OUTFLOW		
            elif bndry[f] == _CIRCLE and mp[0] > gX and mp[1] < gY + gEH + DOLFIN_EPS \
                    and mp[1] > gY - gEH - DOLFIN_EPS:
                unelastic_surface[f] = 1
            else:
                if bndry[f] != _CIRCLE:
                    raise ValueError('Unclassified exterior facet with midpoint [%.3f, %.3f].' \
                        % (mp.x(), mp.y()))
        else:
            flag = 0
            for c in cells(f):
                if domains[c] == 0:
                    flag = flag | 1
                if domains[c] == 1:
                    flag = flag | 2
            if flag == 3:
                interface[f] = 1

    info("\t -done") 
    return(mesh, bndry, interface, unelastic_surface, domains, A, B)

def give_marked_multimesh(background_coarseness = 40, elasticity_coarseness = 40, refine = False):
    # generate multimesh
    approx_circle_with_edges = 20
    beam_mesh_legth = g_radius + 1.5*gEL
    beam_mesh_width = 2.0*g_radius
    beam_mesh_X = gX
    beam_mesh_Y = gY - 2.0*g_radius


    multimesh = MultiMesh()
    bg_geometry = mshr.Rectangle(Point(0.0, 0.0), Point(gL, gW)) - \
                     mshr.Circle(Point(gX, gY), g_radius, approx_circle_with_edges)
    bg_mesh = mshr.generate_mesh(bg_geometry, bg_h)
    if refine:
        parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        for k in range(2):		# 2
            cf = MeshFunction('bool', bg_mesh, 2)
            cf.set_all(False)
            z = 0.08
            for c in cells(bg_mesh):
                if c.midpoint().distance(Point(gX, gY)) < (0.05 + k*z) : cf[c]=True
                elif (c.midpoint()[0] <= gX + g_radius + gEL + k*z and c.midpoint()[0] >= gX - k*z\
                 and c.midpoint()[1] <= gY + 0.5*gEH + k*z and c.midpoint()[1] >= gY - 0.5*gEH - k*z):
                    cf[c] = True
            bg_mesh = refine(bg_mesh, cf)

    multimesh.add(bg_mesh)

    beam_geometry = mshr.Rectangle(Point(beam_mesh_X,beam_mesh_Y ), \
                Point(beam_mesh_X + beam_mesh_length, beam_mesh_Y + baem_mesh_width)) \
                - mshr.Circle(Point(gX, gY), g_radius, approx_circle_with_edges)
    beam_mesh = mshr.generate_mesh(beam_geometry, beam_h)
    multimesh.add(beam_mesh)
    multimesh.build()

    # define boundary and subdomain functions
    class Cylinder(SubDomain):
        def snap(self, x):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            if r <= g_radius:
                x[0] = gX + (g_radius/r) * (x[0] - gX)
                x[1] = gY + (g_radius/r) * (x[1] - gY)
        def inside(self, x, on_boundary):
            r = sqrt((x[0] - gX)**2 + (x[1] - gY)**2)
            return( (r <= g_radius + DOLFIN_EPS) and on_boundary)

    cylinder = Cylinder()

    class Inflow(SubDomain):
        def inside(self, x, on_boundary):
            return(on_boundary and near(x[0], 0.0))

    inflow_bndry = Inflow()

    class Outflow(SubDomain):
        def inside(self, x, on_boundary):
            return(on_boundary and near(x[0], gL))

    outflow_bndry = Outflow()

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return(on_boundary and (near(x[1], 0.0) or near(x[1], gW)))

    walls = Walls()

    class Elasticity(SubDomain):
        def inside(self, x, on_boundary):
            return(x[0] <= gX + g_radius + gEL + DOLFIN_EPS and x[0] >= gX - DOLFIN_EPS\
             and x[1] <= gY + 0.5*gEH + DOLFIN_EPS and x[1] >= gY - 0.5*gEH - DOLFIN_EPS)

    elasticity = Elasticity()

    class ALE_Fluid(SubDomain):
        def inside(self, x, on_boundary):
            return(x[0] >= beam_mesh_X - DOLFIN_EPS \
                    and x[0] <= beam_mesh_X + beam_mesh_length + DOLFIN_EPS\
                    and x[1] >= beam_mesh_Y - DOLFIN_EPS \
                    and x[1] <= beam_mesh_Y + beam_mesh_width + DOLFIN_EPS \
                    and not (x[0] <= gX + g_radius + gEL + DOLFIN_EPS and x[0] >= gX - DOLFIN_EPS\
                      and x[1] <= gY + 0.5*gEH + DOLFIN_EPS and x[1] >= gY - 0.5*gEH - DOLFIN_EPS
                    ))

    ale_fluid = ALE_Fluid()

    ALE_domains = MeshFunction('size_t', multimesh.part(1), 1)
    ALE_domains.set_all(0)
    elasticity.mark(ALE_domains, 1)

    Eulerian_fluid = MeshFunction('size_t', multimesh.part(0), 1)
    Eulerian_fluid.set_all(0)
    

    return(multimesh, inflow_bndry, outflow_bndry, walls, cylinder, ALE_domains, Eulerian_fluid, A, B)

    '''
    # construct facet and domain markers
    interface         = MeshFunction('size_t', beam_mesh, 1)	# interface marker
    unelastic_surface = MeshFunction('size_t', beam_mesh, 1)	# circle surface neighbouring with fluid (not with elastic solid)
    domains           = MeshFunction('size_t', beam_mesh, 2, beam_mesh.domains())
    interface.set_all(0)
    unelastic_surface.set_all(0)
    domains.set_all(0)
    elasticity.mark(domains, 1)
    cylinder.mark(bndry, _CIRCLE)
    for f in facets(beam_mesh):
        if f.exterior():
            mp = f.midpoint()
            if near(mp[0], 0.0):
                bndry[f] = _INFLOW 				
            elif near(mp[1], 0.0) or near(mp[1], gW):
                bndry[f] = _WALLS			
            elif near(mp[0], gL):
                bndry[f] = _OUTFLOW		
            elif bndry[f] == _CIRCLE and mp[0] > gX and mp[1] < gY + gEH + DOLFIN_EPS and mp[1] > gY - gEH - DOLFIN_EPS:
                unelastic_surface[f] = 1
        else:
            flag = 0
            for c in cells(f):
                if domains[c] == 0:
                    flag = flag | 1
                if domains[c] == 1:
                    flag = flag | 2
            if flag == 3:
                interface[f] = 1
    '''

    info("\t\t-done") 
    return(multimesh, inflow_bndry, outflow_bndry, walls, cylinder)#, interface, 
     #       unelastic_surface, domains, A, B)

