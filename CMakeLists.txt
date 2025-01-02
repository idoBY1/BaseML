# CMakeList.txt : CMake project for BasicNeuralNetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.12)

project ("BasicNeuralNetwork")

add_library(MachineLearning)

# folder of include files
target_include_directories(MachineLearning PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

# add all cpp files in src as source code files for this program
file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
target_sources(MachineLearning PRIVATE "${MY_SOURCES}")

set_property(TARGET MachineLearning PROPERTY CXX_STANDARD 20)

# add test file
add_executable(nn_test "nn_test.cpp")
target_link_libraries(nn_test PUBLIC MachineLearning)

set_property(TARGET nn_test PROPERTY CXX_STANDARD 20)
