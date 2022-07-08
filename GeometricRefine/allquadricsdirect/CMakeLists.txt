include_directories(src ${CMAKE_SOURCE_DIR}/GeometricRefine/src include lib/jama125 lib/tnt_126 lib/CLAPACK/include)
FILE(GLOB_RECURSE project_headers src/*.h)
FILE(GLOB_RECURSE project_cpps src/*.cpp)
set(all_files ${project_headers} ${project_cpps})

#windows version
LINK_DIRECTORIES(lib/CLAPACK/lib_64)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_SOURCE_DIR}/Lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_SOURCE_DIR}/Lib)
set(CMAKE_DEBUG_POSTFIX "d")
ADD_LIBRARY(allquadricsdirect STATIC ${all_files})

# set_target_properties(allquadricsdirect PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(allquadricsdirect PUBLIC debug libf2cd debug clapackd debug BLASd optimized libf2c optimized clapack optimized BLAS)