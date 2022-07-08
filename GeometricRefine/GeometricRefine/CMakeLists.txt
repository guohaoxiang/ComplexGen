cmake_minimum_required(VERSION 3.0.0)

project(GeometricRefine VERSION 0.1.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 11)

IF(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
ENDIF()

# find_package(EIGEN REQUIRED)

if(${MSVC})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_SOURCE_DIR}/Bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_SOURCE_DIR}/Bin)
elseif(${CYGWIN})
SET(EXECUTABLE_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/Bin)
elseif(${MINGW})
SET(EXECUTABLE_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/Bin)
elseif(${UNIX})
SET(EXECUTABLE_OUTPUT_PATH  ${CMAKE_SOURCE_DIR}/Bin)
endif(${MSVC})

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/Lib)

LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/allquadricsdirect/lib/CLAPACK/lib_64)

FILE(GLOB_RECURSE project_cpps src/HOptimizer/*.cpp src/CurveFitter.cpp src/SurfFitter.cpp src/Helper.cpp src/main.cpp src/Helper.h src/SurfFitter.h src/CurveFitter.h src/MyPointCloud.h src/KD_Tree.h src/NURBSFitting.h src/NURBSFittingWrapper.h src/NURBSFittingWrapper.cpp src/HOptimizer src/Mesh3D.h src/Mesh3D.cpp)
INCLUDE_DIRECTORIES(src src/HOptimizer src/yaml-cpp)

set(CMAKE_DEBUG_POSTFIX "d")
add_definitions(/bigobj)
add_executable(GeometricRefine ${project_cpps})
# target_link_libraries(GeometricRefine PRIVATE debug allquadricsdirectd debug yaml-cppd optimized allquadricsdirect optimized yaml-cpp legacy_stdio_definitions)
target_link_libraries(GeometricRefine PUBLIC debug allquadricsdirectd debug yaml-cppd optimized allquadricsdirect optimized yaml-cpp legacy_stdio_definitions)


