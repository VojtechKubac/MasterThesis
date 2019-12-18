//SetFactory("OpenCASCADE");

h=0.05;
myhext=h;
//myhprecis=0.003;
myhprecis=myhext/7.;

L = 2.5;
W = 0.41;
cx = 0.2;
cy = 0.2;
r = 0.05;

Point(1) = {0., 0., 0., myhext};
Point(2) = {L, 0., 0., myhext};
Point(3) = {L, W, 0., myhext};
Point(4) = {0., W, 0., myhext};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Point(5) = {cx, cy, 0., myhprecis};
Point(6) = {cx-r, cy, 0., myhprecis};
Point(7) = {cx+r, cy, 0., myhprecis};
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 6};

Point(8) = {0.6, 0.2, 0., myhprecis};
Point(9) = {1.5, 0.2, 0., myhprecis*3.};


// refinement lines
Line(7) = {7, 8};
Line(8) = {8, 9};

// surface fluid
Line Loop(11) = {3, 4, 1, 2};
Line Loop(12) = {5, 6, 7, 8, -8, -7};
Plane Surface(1) = {11, 12};

// refinement points
extreme_precise = 0.015;
Point(10) = {0.44, 0.3, 0., extreme_precise};
Point{10} In Surface{1};
Point(11) = {0.27, 0.27, 0., extreme_precise};
Point{11} In Surface{1};
Point(12) = {0.6, 0.3, 0., extreme_precise};
Point{12} In Surface{1};
Point(13) = {0.44, 0.1, 0., extreme_precise};
Point{13} In Surface{1};
Point(14) = {0.27, 0.13, 0., extreme_precise};
Point{14} In Surface{1};
Point(15) = {0.6, 0.1, 0., extreme_precise};
Point{15} In Surface{1};

Point(16) = {0.52, 0.25, 0., extreme_precise};
Point{16} In Surface{1};
Point(17) = {0.52, 0.15, 0., extreme_precise};
Point{17} In Surface{1};


Physical Line("1") = {4};            // inflow
Physical Line("2") = {1,3};          // rigid walls
Physical Line("3") = {5, 6};         // whole circle
Physical Line("4") = {2};            // outflow

Physical Surface("1") = {1};


Mesh.Algorithm = 8; // Frontal
Mesh.Optimize=1;
Mesh.OptimizeThreshold=0.5;
Mesh.AnisoMax=10.0;
Mesh.Smoothing=100;
Mesh.OptimizeNetgen=1;

Mesh 2;

If ( levels > 0 )
For i In { 1 : levels }
  RefineMesh;
EndFor 
EndIf


