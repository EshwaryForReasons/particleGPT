cmake_minimum_required(VERSION 3.20)

set( CMAKE_CXX_STANDARD 23 )
set(CMAKE_CXX_FLAGS "-O3")

project(pTokenizer)

add_executable(pTokenizer main.cpp)
target_include_directories(pTokenizer PUBLIC)
target_link_libraries(pTokenizer PUBLIC)