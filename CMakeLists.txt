set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/lib)

cmake_minimum_required(VERSION 3.10)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

enable_language (C CXX)
enable_testing ()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 -DDEBUG")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(MPI)
if (MPI_FOUND)
  include_directories(${MPI_C_INCLUDE_PATH})
  link_libraries(${MPI_C_LIBRARIES})
  link_libraries(${MPI_CXX_LIBRARIES})
endif()

find_package(GSL)
if (GSL_FOUND)
  link_libraries(${GSL_LIBRARIES})
  include_directories(${GSL_INCLUDE_DIR})
endif()

find_package(CBLAS)
if (CBLAS_FOUND)
  include_directories(${CBLAS_INCLUDE_DIR})
  add_definitions(-DCBLAS=<${CBLAS_INCLUDE_FILE}>)
  link_libraries(${BLAS_LIBRARIES})
elseif()
  message("CBLAS NOT FOUND")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(ARGP)
if (ARGP_FOUND)
  include_directories(${ARGP_INCLUDE_DIRS})
  link_libraries(${ARGP_LIBRARIES})
endif()

project (ccd)

add_subdirectory(src)


message("BLAS LIBS: ${BLAS_LIBRARIES}")
message("CBLAS INCLUDE DIR: ${CBLAS_INCLUDE_DIR}")
message("CBLAS FILE: ${CBLAS_INCLUDE_FILE}")
message("CBLAS LIBS: ${CBLAS_LIBRARIES}")
message("MPI INC: ${MPI_C_INCLUDE_PATH}")
message("MPI LIB: ${MPI_CXX_LIBRARIES}")
message("GSL INC: ${GSL_INCLUDE_DIR}")
message("GSL LIB: ${GSL_LIBRARIES}")


