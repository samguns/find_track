cmake_minimum_required(VERSION 3.5)
project(find_track)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(${PROJECT_NAME} main.cpp utils.cpp tracker.cpp tracker.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

add_executable(play play_ground.cpp utils.cpp tracker.cpp tracker.h)
target_link_libraries(play ${OpenCV_LIBS})
target_link_libraries (play Eigen3::Eigen)
