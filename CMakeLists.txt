cmake_minimum_required(VERSION 3.14)
project(bit-depth-adc-filter)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main pybind11::embed)
