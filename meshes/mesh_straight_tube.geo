//SetFactory("OpenCASCADE");

r = 1.0;	// inner radius
R = 1.1;	// outer radius
H = 12.0;	// height of cylinder

h = 0.2;		// coarseness
myhprecis = h/5.0;	// finer coarseness
layers = 8*H;		// number of layers along cylinder

Point(1) = {0.0, 0.0, 0.0, h};		// origin
Point(2) = {0.0, r, 0.0, h};		// point on inner circle
Point(3) = {0.0, 0.0, r, h};		// point on inner circle
Point(4) = {0.0, -r, 0.0, h};		// point on inner circle
Point(5) = {0.0, 0.0, -r, h};		// point on inner circle
Point(6) = {0.0, R, 0.0, h};		// point on outer circle
Point(7) = {0.0, 0.0, R, h};		// point on outer circle
Point(8) = {0.0, -R, 0.0, h};		// point on outer circle
Point(9) = {0.0, 0.0, -R, h};		// point on outer circle
Point(10) = {H, 0.0, 0.0, h};		// center of the top circles

// inner circle
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

// outer circle
Circle(5) = {6, 1, 7};
Circle(6) = {7, 1, 8};
Circle(7) = {8, 1, 9};
Circle(8) = {9, 1, 6};

Line Loop(9)  = {1, 2, 3, 4};
Line Loop(10) = {5, 6, 7, 8};

Plane Surface(1) = {9};		// surface of inner circle
Plane Surface(2) = {10, 9};	// surface of ring

Line(11) = {1, 10};		// connects centres of circles

//extrusion[] = Extrude{ Point{10}; }{ Surface{1}; };
circle_extrusion[] = Extrude{H, 0.0, 0.0}{
			Surface{1}; 
			Layers{layers};
			QuadTriAddVerts;
		};
ring_extrusion[] = Extrude{H, 0.0, 0.0}{ 
			Surface{2}; 
			Layers{layers};
			QuadTriAddVerts;
		};

//Transfinite Surface(99) = ring_extrusion[2];

Physical Volume(0) = circle_extrusion[1];	// fluid domain
Physical Volume(1) = ring_extrusion[1];		// solid domain

Physical Surface(1) = {1};			// inflow - fluid part
Physical Surface(2) = {2};			// inflow - solid part
Physical Surface(3) = circle_extrusion[0];	// outflow - fluid part
Physical Surface(4) = ring_extrusion[0];	// outflow - solid part


// meshing
Mesh.Algorithm = 8; // Frontal
Mesh.Optimize=1;
Mesh.OptimizeThreshold=0.5;
Mesh.AnisoMax=10.0;
Mesh.Smoothing=100;
Mesh.OptimizeNetgen=1;

Mesh 3;

If ( levels > 0 )
For i In { 1 : levels }
  RefineMesh;
EndFor 
EndIf

