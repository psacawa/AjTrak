cmake_minimum_required(VERSION 2.8)
project( AjTrak )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set (CMAKE_CXX_STANDARD 11)
SET(GCC_COVERAGE_COMPILE_FLAGS "-Wall")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# plik wykonywalny
add_executable( ObjectDetection ObjectDetection.cpp )
add_executable( ImageCapture ImageCapture.cpp )
add_executable( TrainNetwork TrainNetwork.cpp )

add_executable( CVTest CVTest.cpp )
add_executable( Aux Aux.cpp )
add_executable( MaskTest MaskTest.cpp )
add_executable( PupilDetect PupilDetect.cpp )

# łączenie z bibliotekami
target_link_libraries( ObjectDetection ${OpenCV_LIBS} )
target_link_libraries( ImageCapture ${OpenCV_LIBS} )
target_link_libraries( TrainNetwork ${OpenCV_LIBS} )
target_link_libraries( CVTest ${OpenCV_LIBS} )
target_link_libraries( ImageCapture xdo )
target_link_libraries( MaskTest  ${OpenCV_LIBS} )
target_link_libraries( PupilDetect  ${OpenCV_LIBS} )

target_link_libraries( Aux xdo )
