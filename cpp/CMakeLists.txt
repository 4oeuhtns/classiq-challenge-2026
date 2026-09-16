cmake_minimum_required(VERSION 3.15)
project(oracle_synth LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CATERPILLAR_TEST OFF CACHE BOOL "" FORCE)
set(CATERPILLAR_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CATERPILLAR_Z3 OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/caterpillar)

add_subdirectory(src)