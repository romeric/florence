project(PlanarMeshGenerator)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -mavx -Wall -Wextra")
include_directories(${/home/roman/Dropbox/eigen-devel/})

