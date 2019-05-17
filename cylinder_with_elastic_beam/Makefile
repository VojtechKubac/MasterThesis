#!/usr/bin/env gmake
SHELL=/bin/bash
.SECONDARY:

DATA:=meshes

all: mesh_files

mesh_files: $(DATA)/mesh1.h5 $(DATA)/mesh1_refined.h5 $(DATA)/mesh2.h5

$(DATA):
	mkdir -p $(DATA)

$(DATA)/mesh1.h5: $(DATA)/mesh1.xml
	python3 xml2h5.py $(DATA)/mesh1

$(DATA)/mesh1_refined.h5: $(DATA)/mesh1_refined.xml
	python3 xml2h5.py $(DATA)/mesh1_refined

$(DATA)/mesh2.h5: $(DATA)/mesh2.xml
	python3 xml2h5.py $(DATA)/mesh2

$(DATA)/%: $(DATA)/%.h5
	echo $< $@ $* $^


$(DATA)/%.xml : $(DATA)/%.msh
	dolfin-convert $< $@

$(DATA)/%.msh: %.geo | $(DATA)
	gmsh -2 $< -o $@ -format msh2


clean:
	rm  $(DATA)/*.xml
	#rm  $(DATA)/*.h5
	#rm -rf $(DATA)
