# Copyright (c) 2025 Mikko Kuha (University of Jyväskylä).
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details. You should have received a copy
# of the GNU General Public License along with this program. If not, see
# <http://www.gnu.org/licenses/>.
#
# BSD 3-clause license, see LICENSE.txt

#IF lhapdf-config DOESN'T WORK FOR YOU, CHECK THE TWO COMMENTS BELOW
add_library(
	Parts ars.cpp
	hcubature.cpp
	histo.cpp
	linear_interpolator.cpp
	mcaa.cpp
	nn_coll.cpp
	nucleus_generator.cpp
	pqcd.cpp
	eps09.cxx
)

execute_process(
	COMMAND bash -c "lhapdf-config --includedir"
	OUTPUT_VARIABLE lhapdf_incdir
)
string(STRIP ${lhapdf_incdir} lhapdf_incdir)

execute_process(
	COMMAND bash -c "lhapdf-config --libdir"
	OUTPUT_VARIABLE lhapdf_libdir
) 
string(STRIP ${lhapdf_libdir} lhapdf_libdir)

target_include_directories(Parts PUBLIC 
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${lhapdf_incdir}" #REPLACE THIS WITH THE PATH TO THE DIRECTORY OF LHAPDF HEADER FILES (.h)!!!
)

target_compile_features(Parts PUBLIC cxx_std_20)

target_compile_options(Parts PUBLIC
	-Wall
	-fconcepts #COMMENT THIS IF YOU USE CLANG
	-fopenmp
	#-fsanitize=undefined
	#-fsanitize=address
	-O2
)

find_package(Boost REQUIRED)
find_package(GSL REQUIRED)

target_link_directories(Parts PUBLIC 
	"${lhapdf_libdir}" #REPLACE THIS WITH THE PATH TO THE DIRECTORY OF LHAPDF LIBRARY FILES (.so)!!!
)

target_link_libraries(Parts PUBLIC 
	GSL::gsl
	GSL::gslcblas
	-fopenmp
	#-fsanitize=undefined
	#-fsanitize=address
	-lpthread
	-lm
	-lLHAPDF
)
