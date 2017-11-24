cmake_minimum_required(VERSION 2.8)
project( ObjectDetection )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set (CMAKE_CXX_STANDARD 11)
SET(GCC_COVERAGE_COMPILE_FLAGS "-Wall")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

add_executable( ObjectDetection ObjectDetection.cpp )

target_link_libraries( ObjectDetection ${OpenCV_LIBS} )
