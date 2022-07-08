find_package(OpenMP REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}")
FILE(GLOB_RECURSE project_headers *.h)
FILE(GLOB_RECURSE project_cpps *.cpp)
set(all_files ${project_headers} ${project_cpps})
source_group_by_dir(all_files)

if(${MSVC}) 
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE  ${PROJECT_SOURCE_DIR}/HandyMathLib/msvc64)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG  ${PROJECT_SOURCE_DIR}/HandyMathLib/msvc64)
elseif(${CYGWIN})
SET(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/HandyMathLib/cygwin64)
elseif(${MINGW})
SET(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/HandyMathLib/mingw64)
elseif(${UNIX})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/HandyMathLib/linux64)
endif(${MSVC})



set(CMAKE_DEBUG_POSTFIX "d")
ADD_LIBRARY(HOptimizer STATIC ${all_files})
set_target_properties(HOptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(HOptimizer PRIVATE OpenMP::OpenMP_CXX)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/HOptimizer.h ${CMAKE_CURRENT_SOURCE_DIR}/HLBFGS_Funcname.h ${CMAKE_CURRENT_SOURCE_DIR}/HLBFGS_Sparse_Matrix.h ${CMAKE_CURRENT_SOURCE_DIR}/HOptimizer_Internal.h ${CMAKE_CURRENT_SOURCE_DIR}/HLBFGS_Constraint.h
     DESTINATION ${PROJECT_SOURCE_DIR}/HandyMathInc/HOptimizer)


