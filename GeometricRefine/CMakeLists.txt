cmake_minimum_required(VERSION 3.12)

project(GeometricRefine)
add_subdirectory(${CMAKE_SOURCE_DIR}/allquadricsdirect)
add_subdirectory(${CMAKE_SOURCE_DIR}/yaml-cpp)
include_directories(${CMAKE_SOURCE_DIR}/yaml-cpp/include/yaml-cpp)
include_directories(${CMAKE_SOURCE_DIR}/eigen3)
add_subdirectory(${CMAKE_SOURCE_DIR}/GeometricRefine)
