cmake_minimum_required(VERSION 3.5)

project( PainterlyRendering )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( PainterlyRendering PainterlyRendering.cpp )

target_link_libraries( PainterlyRendering ${OpenCV_LIBS} )