#!/usr/bin/env gmake
SHELL=/bin/bash
.SECONDARY:

NAME1:=mesh_ALE
NAME2:=mesh_Euler
NAME3:=mesh_bench3D
NAME4:=mesh_straight_tube
NAME5:=mesh_bent_tube

GMSH:=gmsh
PYTHON:=python3

all: meshes

meshes: $(NAME1)_L0  $(NAME1)_L1  $(NAME1)_L2  #  $(NAME1)_L3 $(NAME1)_L4
meshes: $(NAME2)_L0  $(NAME2)_L1  $(NAME2)_L2  #  $(NAME2)_L3 $(NAME2)_L4
meshes: $(NAME3)_L0 # $(NAME3)_L1  $(NAME3)_L2    
meshes: $(NAME4)_L0 # $(NAME3)_L1  $(NAME3)_L2    
meshes: $(NAME5)

$(NAME1)%: $(NAME1)%.h5
	echo $< $@ $* $^

$(NAME2)%: $(NAME2)%.h5
	echo $< $@ $* $^

$(NAME3)%: $(NAME3)%.h5
	echo $< $@ $* $^

$(NAME4)%: $(NAME4)%.h5
	echo $< $@ $* $^

%.h5: %.xml 
	$(PYTHON) mesh_process.py --mesh $*

%_L0.msh: %.geo
	$(GMSH) -format msh2 -3 -setnumber levels 0 -o $@ $< 

%_L1.msh: %.geo
	$(GMSH) -format msh2 -3 -setnumber levels 1 -o $@ $< 

%_L2.msh: %.geo
	$(GMSH) -format msh2 -3 -setnumber levels 2 -o $@ $< 

%_L3.msh: %.geo
	$(GMSH) -format msh2 -3 -setnumber levels 3 -o $@ $< 

%_L4.msh: %.geo
	$(GMSH) -format msh2 -3 -setnumber levels 4 -o $@ $< 

%.xml : %.msh 
	dolfin-convert $< $@

$(NAME5)%: $(NAME5)%.xml
	echo $< $@ $* $^
	$(PYTHON) process_salome.py --mesh %*

#%.xml %_physical_region.xml %_facet_region.xml : %.msh
#	dolfin-convert $*.msh $*.xml

clean:
	rm *.msh
	rm $(NAME1)*.xml
	rm $(NAME2)*.xml
	rm $(NAME3)*.xml
	rm $(NAME4)*.xml
	rm *.xdmf
	#rm mesh_ALE_L?.h5
#	rm -rf $(DATA)
