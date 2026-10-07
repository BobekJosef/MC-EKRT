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
cmake_minimum_required(VERSION 3.10)

project(mcaa CXX)

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

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

set( CMAKE_VERBOSE_MAKEFILE on )

add_subdirectory(src)

add_executable(mcaa main.cpp)

target_compile_features(mcaa PUBLIC cxx_std_20)

target_compile_options(mcaa PUBLIC
	-Wall
	-fconcepts #COMMENT THIS IF YOU USE CLANG
	-fopenmp
	#-fsanitize=undefined
	#-fsanitize=address
	-O2
)

target_include_directories(mcaa PUBLIC 
	"${PROJECT_BINARY_DIR}"
	"${lhapdf_incdir}" #REPLACE THIS WITH THE PATH TO THE DIRECTORY OF LHAPDF HEADER FILES (.h)!!!
)

find_package(Boost REQUIRED)
find_package(GSL REQUIRED)

target_link_directories(mcaa PUBLIC 
	"${lhapdf_libdir}" #REPLACE THIS WITH THE PATH TO THE DIRECTORY OF LHAPDF LIBRARY FILES (.so)!!!
)

target_link_libraries(mcaa PUBLIC 
	GSL::gsl
	GSL::gslcblas
	-fopenmp
	-lpthread
	#-fsanitize=undefined
	#-fsanitize=address
	-lm
	-lLHAPDF
	Parts
)

# Converter of the end-state minijets into T^{mu nu}(tau0) for KoMPoST3D (see ekrt_to_kompost.cpp)
add_executable(ekrt_to_kompost ekrt_to_kompost.cpp)
target_compile_features(ekrt_to_kompost PUBLIC cxx_std_17)
target_compile_options(ekrt_to_kompost PUBLIC -Wall -O2)
