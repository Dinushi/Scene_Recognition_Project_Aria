cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


set(CMAKE_CXX_STANDARD 17)

project(ply_generator)

find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (ply_generator ply_generator.cpp)
target_link_libraries (ply_generator ${PCL_LIBRARIES})
