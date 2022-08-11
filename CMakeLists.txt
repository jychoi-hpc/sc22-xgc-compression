cmake_minimum_required (VERSION 3.20)

project (xgc-compression-proxy)

find_package (MPI COMPONENTS CXX)
find_package (ADIOS2 REQUIRED COMPONENTS CXX MPI)
find_package (OpenMP)

add_executable (xgc_proxy xgc_proxy.cpp)
target_link_libraries (xgc_proxy adios2::adios2 MPI::MPI_C)

add_executable (xgc_fdata_reader xgc_fdata_reader.cpp)
target_link_libraries (xgc_fdata_reader adios2::adios2 MPI::MPI_C)

add_executable (adios_test adios_test.cpp)
target_link_libraries (adios_test adios2::adios2 MPI::MPI_C)

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})
target_link_libraries(xgc_proxy ${YAML_CPP_LIBRARIES})