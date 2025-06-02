cmake_minimum_required(VERSION 3.20)

set( CMAKE_CXX_STANDARD 23 )
set( CMAKE_CXX_FLAGS "-O3 --std=c++23" )

project(pTokenizerModule)

find_package(ROOT REQUIRED COMPONENTS RIO Net Hist MathCore Physics)
include(${ROOT_USE_FILE})

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_subdirectory(pybind11)

pybind11_add_module(pTokenizerModule main.cpp tokenizer.cpp untokenizer.cpp)
target_include_directories(pTokenizerModule PUBLIC . pybind11/include ${ROOT_INCLUDE_DIRS})
target_link_libraries(pTokenizerModule PUBLIC ${ROOT_LIBRARIES} EG ${Geant4_LIBRARIES})
target_compile_definitions(pTokenizerModule PRIVATE ${ROOT_DEFINITIONS})