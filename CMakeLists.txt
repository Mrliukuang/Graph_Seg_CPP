cmake_minimum_required(VERSION 3.2)
project(Graph_Seg)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(Graph_Seg ${SOURCE_FILES} main.cpp UF.h)

find_package(OpenCV REQUIRED)
target_link_libraries(Graph_Seg ${OpenCV_LIBS})