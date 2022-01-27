cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(cpl_cu_icp LANGUAGES CXX CUDA)

enable_language(CUDA)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 61 75)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----- System packages

# apt: libgflags-dev
find_package(gflags REQUIRED)

# apt: libopencv-dev
find_package(OpenCV REQUIRED)

# apt: libboost-all-dev
find_package(Boost REQUIRED)

# apt: python3.x-dev
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ----- External libs

add_subdirectory(external)

# ----- Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
# set(CUDA_SEPARABLE_COMPILATION ON)

# build the pymod in the python/external folder to be used by python code
set(PYMOD_OUTPUT "${CMAKE_SOURCE_DIR}/bin")

# set(CMAKE_CUDA_FLAGS_RELEASE "-std=c++17 -O3 -DNDEBUG --generate-line-info")
# set(CMAKE_CUDA_FLAGS_DEBUG "-std=c++17 -G -g")


#------------------- CORE -------------------
add_library(core_lib INTERFACE)
target_include_directories(core_lib INTERFACE include)


#------------------- OCTREE -------------------
add_library(cuda_octree_lib INTERFACE)
target_include_directories(cuda_octree_lib INTERFACE include)
target_link_libraries(cuda_octree_lib INTERFACE cudex_lib core_lib eigen ext_octree)


#------------------- ICP -------------------
add_library(icp_lib STATIC src/icp.cpp.cu)
target_include_directories(icp_lib PUBLIC include)
target_link_libraries(icp_lib PUBLIC cudex_lib core_lib	cuda_octree_lib eigen)


#------------------- Python binding -------------------
pybind11_add_module(pyircp NO_EXTRAS src/pyircp.cpp src/icp_wrapper.cpp.cu)
target_link_libraries(pyircp PUBLIC icp_lib)

#------------------- TESTS -------------------
FILE(GLOB TESTS_SOURCE ${CMAKE_SOURCE_DIR}/tests/*.cu*)
add_executable(unit_tests ${TESTS_SOURCE})

target_link_libraries(unit_tests core_lib cuda_octree_lib icp_lib gtest gtest_main)
add_test( unit_tests unit_tests )
