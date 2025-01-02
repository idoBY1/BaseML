# CMakeList.txt : CMake project for BasicNeuralNetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("BasicNeuralNetwork")

# add all cpp files in src as source code files for this program
file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable (nn_test "nn_test.cpp" "${MY_SOURCES}")

# path to resources (uncomment the right definition for development or release, only one should be uncommented at a time)
#target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/") # for development
#target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC RESOURCES_PATH="./resources/") # for release

# folder of include files
target_include_directories(nn_test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET nn_test PROPERTY CXX_STANDARD 20)
endif()

