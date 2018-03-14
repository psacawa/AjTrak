cmake_minimum_required(VERSION 2.8)
project( AjTrak )

find_package( PkgConfig REQUIRED )
find_package( OpenCV REQUIRED )
find_package(Qt5Widgets REQUIRED )
pkg_search_module( cairo REQUIRED cairo )
# find_package( cairo  REQUIRED )
# message (${OpenCV_INCLUDE_DIRS})
# message (${OpenCV_LIBS})

# add_library ( tensorflow_all STATIC IMPORTED )
# set_property ( TARGET tensorflow_all PROPERTY IMPORTED_LOCATION
#	 /usr/local/lib/libtensorflow_all.so )
# add_library ( tensorflow_framework STATIC IMPORTED )
# set_property ( TARGET tensorflow_framework PROPERTY IMPORTED_LOCATION
#     /usr/local/lib/libtensorflow_framework.so )

# SET(CMAKE_CXX_COMPILER  $(which g++-6))

include_directories( include/eigen3 )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( /usr/local/include/google/tensorflow )
include_directories( ${CMAKE_SOURCE_DIR}/include )
include_directories(${Qt5Widgets_INCLUDE_DIRS})
# set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set (CMAKE_CXX_STANDARD 11)
SET(GCC_COVERAGE_COMPILE_FLAGS "-Wall")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# plik wykonywalny
add_executable( ObjectDetection ObjectDetection.cpp )
add_executable( ImageCapture ImageCapture.cpp )
add_executable( TensorflowTest TensorflowTest.cpp )
add_executable( CairoTest CairoTest.cpp )
# add_executable( TrainNetwork TrainNetwork.cpp )

# łączenie z bibliotekami
target_link_libraries( ObjectDetection ${OpenCV_LIBS} )
target_link_libraries( ImageCapture ${OpenCV_LIBS} )

target_link_libraries( TensorflowTest tensorflow_cc)
# target_link_libraries( ImageCapture tensorflow_all)
# target_link_libraries( TensorflowTest tensorflow_framework)
target_link_libraries( ImageCapture xdo )
target_link_libraries( CairoTest cairo)
target_link_libraries( CairoTest Qt5Widgets )
