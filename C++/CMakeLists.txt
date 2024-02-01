cmake_minimum_required(VERSION 3.27)
project(Projects_MPI)

enable_language(CXX)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FILES_1 minmax.cpp)
add_executable(openmpi_minmax ${SOURCE_FILES_1})
target_link_libraries(openmpi_minmax ${MPI_CXX_LIBRARIES})

set(SOURCE_FILES_2 scalar.cpp)
add_executable(openmpi_scalar ${SOURCE_FILES_2})
target_link_libraries(openmpi_scalar ${MPI_CXX_LIBRARIES})

set(SOURCE_FILES_3 message.cpp)
add_executable(openmpi_message ${SOURCE_FILES_3})
target_link_libraries(openmpi_message ${MPI_CXX_LIBRARIES})

set(SOURCE_FILES_4 matrixStrip.cpp)
add_executable(openmpi_matrixStrip ${SOURCE_FILES_4})
target_link_libraries(openmpi_matrixStrip ${MPI_CXX_LIBRARIES})

set(SOURCE_FILES_5 matrixFox.cpp)
add_executable(openmpi_matrixFox ${SOURCE_FILES_5})
target_link_libraries(openmpi_matrixFox ${MPI_CXX_LIBRARIES})

