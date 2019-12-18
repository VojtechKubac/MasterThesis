#!/usr/bin/env gmake
SHELL=/bin/bash
.SECONDARY:

NAME1:=mesh_ALE
NAME2:=mesh_Euler
NAME3:=mesh_3Dbench
NAME4:=mesh_straight_tube
NAME5:=mesh_bent_tube

#GMSH:=gmsh
GMSH:=/usr/local/pkg/./Installs/linux-ubuntu16.04-x86_64-gcc7.4.0/bin-gmsh/4.1.5/rcrbli44ahxkxawu/bin/gmsh -nt 8
#PYTHON:=python3
PYTHON:=python

all: 2D 3D

2D: D:=2
3D: D:=3

2D: $(NAME1)_L0  $(NAME1)_L1  $(NAME1)_L2  #  $(NAME1)_L3 $(NAME1)_L4
2D: $(NAME2)_L0  $(NAME2)_L1  $(NAME2)_L2  #  $(NAME2)_L3 $(NAME2)_L4
3D: $(NAME3)_L0 # $(NAME3)_L1  $(NAME3)_L2    
3D: $(NAME4)_L0 # $(NAME3)_L1  $(NAME3)_L2    
3D: $(NAME5)

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
	$(GMSH) -format msh2 -${D} -setnumber levels 0 -o $@ $< 

%_L1.msh: %.geo
	$(GMSH) -format msh2 -${D} -setnumber levels 1 -o $@ $< 

%_L2.msh: %.geo
	$(GMSH) -format msh2 -${D} -setnumber levels 2 -o $@ $< 

%_L3.msh: %.geo
	$(GMSH) -format msh2 -${D} -setnumber levels 3 -o $@ $< 

%_L4.msh: %.geo
	$(GMSH) -format msh2 -${D} -setnumber levels 4 -o $@ $< 

%.xml : %.msh 
	dolfin-convert $< $@

$(NAME5): $(NAME5)_L0.h5
	echo $< $@ $* $^

$(NAME5)_L0.h5: $(NAME5).xml
	$(PYTHON) process_salome.py --mesh $(NAME5)

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
