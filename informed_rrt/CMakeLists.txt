cmake_minimum_required(VERSION 3.1)
project( informed_rrt )
find_package( OpenCV REQUIRED )
find_package( Eigen3 3.1.0 REQUIRED )
include_directories(
  ${OpenCV_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)
add_executable( informed_rrt informed_rrt.cpp )
target_link_libraries( informed_rrt ${OpenCV_LIBS} ${EIGEN3_LIBS})
