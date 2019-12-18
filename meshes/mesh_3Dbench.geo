SetFactory("OpenCASCADE");

N=8; // global mesh density control

r=0.05;		// radius of cylinder

L=2.8;		// length of domain
W=0.41;		// width of doamin

cx=0.5;		// x coordinate of centre of cylinder
cy=0.2;		// y coordinate of centre of cylinder

dd=0.01;	// half-width of beam

refine=0.9; // this much fine mesh around the cylinder

cl1 = W/N;
cl2 = refine*cl1;
dr=0.25*r;
rr=r+dr;     // radius of the finer mesh around cylinder

box = 1;
Box(box)={0,0,0,L,W,W};

beam=2;
Box(beam)={0.5,0.2-dd,0.1, 0.4,2*dd,0.2};

cylinder=3;
Cylinder(cylinder)={cx,cy,0,0,0,W,r};

Cylinder(21)={cx,cy,0,0,0,W,rr};
BooleanDifference(22) = { Volume{21}; Delete; }{ Volume{cylinder}; };

Box(41)={0.5,0.2-2*dd,0.1-dd, 0.4+dd,4*dd,W-0.2+2*dd};
BooleanDifference(42) = { Volume{41}; Delete; }{ Volume{cylinder, beam}; };

BooleanDifference(4) = { Volume{box}; Delete; }{Volume{cylinder, beam}; };

//BooleanIntersection(14) = { Volume{13}; Delete; }{ Volume{4}; };
BooleanDifference(5) = { Volume{beam}; Delete; }{ Volume{cylinder}; Delete; };

SyncModel;

v() = BooleanFragments{ Volume{4,5,22,42}; Delete; }{ };

// gmsh on my laptop
// Physical Surface("1") = {1}; 				// inflow
// Physical Surface("2") = {2,3,4,5,13,18}; 			// wall
// Physical Surface("3") = {14,22,23};				// cylinder/fluid
// Physical Surface("4") = {34,36};				// cylinder/solid
// Physical Surface("5") = {6}; 				// outflow
// Physical Surface("6") = {24,28,30,29,27,26,25,31,32}; 	// FSI

// gmsh on cluster
Physical Surface("1") = {1}; 				// inflow
Physical Surface("2") = {2,3,4,5,13,18}; 		// wall
Physical Surface("3") = {14,27,26};			// cylinder/fluid
Physical Surface("4") = {34,36};			// cylinder/solid
Physical Surface("5") = {6}; 				// outflow
Physical Surface("6") = {24,22,23,28,30,29,25,31,32}; 	// FSI

Physical Volume(0) = {1,2,3,4};		// fluid
Physical Volume(1) = {5,6};		// solid

l() = Unique(Abs(Boundary{ Surface{18,17,16,15,14,13,7,28,32,31,30,29,19,27,21,20}; }));
p() = Unique(Abs(Boundary{ Line{l()}; }));
Characteristic Length{p()} = 2*dr;

l() = Unique(Abs(Boundary{ Surface{12,11,10,9,8,22,35,33,26,25,24,23,21,20}; }));
p() = Unique(Abs(Boundary{ Line{l()}; }));
Characteristic Length{p()} = dd;


Field[2] = Cylinder;
Field[2].VIn = cl2;
Field[2].VOut = cl1;
Field[2].Radius = r+dr;
Field[2].XAxis = 0.0;
Field[2].YAxis = 0.0;
Field[2].ZAxis = W+0.2;
Field[2].XCenter = cx;
Field[2].YCenter = cy;
Field[2].ZCenter = 0.0-0.1;

Field[3] = Cylinder;
Field[3].VIn = cl2;
Field[3].VOut = cl1;
Field[3].Radius = r+dr*0.5;
Field[3].XAxis = 0.0;
Field[3].YAxis = 0.0;
Field[3].ZAxis = W+0.2;
Field[3].XCenter = cx;
Field[3].YCenter = cy;
Field[3].ZCenter = 0.0-0.1;

Field[4] = Box;
Field[4].VIn = 2*dd;
Field[4].VOut = cl1;
Field[4].XMax = cx+0.4;
Field[4].XMin = cx;
Field[4].YMax = cy+dd;
Field[4].YMin = cy-dd;
Field[4].ZMax = W-0.1;
Field[4].ZMin = 0.0+0.1;

Field[5] = Box;
Field[5].VIn = 3*dd;
Field[5].VOut = cl1;
Field[5].XMax = cx+0.4+4*dr;
Field[5].XMin = cx;
Field[5].YMax = cy+dd+8*dr;
Field[5].YMin = cy-dd-8*dr;
Field[5].ZMax = W;
Field[5].ZMin = 0.0;

Field[6] = Box;
Field[6].VIn = 4*dd;
Field[6].VOut = cl1;
Field[6].XMax = cx+1.8;
Field[6].XMin = cx;
Field[6].YMax = W;
Field[6].YMin = 0.0;
Field[6].ZMax = W;
Field[6].ZMin = 0.0;

Field[1] = Min;
Field[1].FieldsList = {2,4,5,6};

Background Field = 1;

//Mesh.RemeshParametrization = 7; // conformal finite element
Mesh.Algorithm = 6; // Frontal
Mesh.Algorithm3D = 2;
//Mesh.Optimize=1;
Mesh.OptimizeThreshold=0.5;
Mesh.AnisoMax=2.0;
Mesh.Smoothing=100;
//Mesh.SmoothNormals=1;
//Mesh.OptimizeNetgen=1;
//Mesh.CharacteristicLengthMin = cl2;
//Mesh.CharacteristicLengthMax = cl1;


Mesh 3;
OptimizeMesh "Gmsh";
//OptimizeMesh "Netgen";
//Save Sprintf("bench3D_fsi_L%02g.msh", 0);
//Save Sprintf("bench3D_fsi_L%02g.vtk", 0);

//For i In {1:2}
//  RefineMesh;
//  OptimizeMesh "Gmsh";
//  //OptimizeMesh "Netgen";
//  //SetOrder 2;
//  Save Sprintf("bench3D_fsi_L%02g.msh", i);
//  Save Sprintf("bench3D_fsi_L%02g.vtk", i);
//EndFor

