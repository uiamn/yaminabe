cmake_minimum_required(VERSION 3.1)
project( rrt )
find_package( OpenCV REQUIRED )
add_executable( rrt rrt.cpp )
target_link_libraries( rrt ${OpenCV_LIBS} )
