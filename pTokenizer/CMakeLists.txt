cmake_minimum_required(VERSION 3.20)

set( CMAKE_CXX_STANDARD 23 )
set( CMAKE_CXX_FLAGS "-O3 --std=c++23" )

project(pTokenizerModule)

add_subdirectory(pybind11)

pybind11_add_module(pTokenizerModule main.cpp dictionary.cpp tokenizer.cpp untokenizer.cpp filter.cpp)
target_include_directories(pTokenizerModule PUBLIC . pybind11/include)