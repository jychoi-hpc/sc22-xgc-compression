cmake_minimum_required (VERSION 3.20)

project (xgc-compression-proxy)

find_package (MPI COMPONENTS CXX)
find_package (ADIOS2 REQUIRED COMPONENTS CXX MPI)
find_package (OpenMP)

add_executable (xgc_writer xgc_writer.cpp)
target_link_libraries (xgc_writer adios2::adios2 MPI::MPI_C)

add_executable (xgc_compress xgc_compress.cpp)
target_link_libraries (xgc_compress adios2::adios2 MPI::MPI_C)

add_executable (xgc_fdata_reader xgc_fdata_reader.cpp)
target_link_libraries (xgc_fdata_reader adios2::adios2 MPI::MPI_C)

add_executable (adios_test adios_test.cpp)
target_link_libraries (adios_test adios2::adios2 MPI::MPI_C)

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})
target_link_libraries(xgc_compress ${YAML_CPP_LIBRARIES})
message(STATUS YAML_CPP_LIBRARIES ${YAML_CPP_LIBRARIES})
message(STATUS YAML_CPP_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIR})

find_path(GPTL_INCLUDE_DIR NAMES gptl.h)
find_library(GPTL_LIBRARY NAMES gptl)
message(STATUS GPTL_LIBRARY:"${GPTL_INCLUDE_DIR}")
message(STATUS GPTL_LIBRARY:"${GPTL_LIBRARY}")
include_directories(${GPTL_INCLUDE_DIR})
target_link_libraries(xgc_compress "${GPTL_LIBRARY}")

find_path(PAPI_INCLUDE_DIR NAMES papi.h)
find_library(PAPI_LIBRARY NAMES papi)
message(STATUS PAPI_LIBRARY:"${PAPI_INCLUDE_DIR}")
message(STATUS PAPI_LIBRARY:"${PAPI_LIBRARY}")
include_directories(${PAPI_INCLUDE_DIR})
target_link_libraries(xgc_compress "${PAPI_LIBRARY}")

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED OFF)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS log log_setup program_options filesystem)
message(STATUS "Boost_INCLUDE_DIRS: " "${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: " "${Boost_LIBRARIES}")
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(xgc_compress ${Boost_LIBRARIES})