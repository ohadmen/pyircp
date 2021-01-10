# ----- gtest, glog
add_subdirectory(googletest)
add_subdirectory(glog)

# ----- cudex
add_subdirectory(cudex)

# ----- pybind
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python executable" FORCE)
add_subdirectory(pybind11)

# ----- eigen
add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE eigen)

# ----- octree

# library
add_library(ext_octree INTERFACE)
target_include_directories(ext_octree INTERFACE octree)

# test
add_executable(ext_octree_test octree/test/octree-test.cpp)
target_link_libraries(ext_octree_test gtest_main)

add_test(ext_octree_test ext_octree_test)

