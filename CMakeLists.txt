cmake_minimum_required(VERSION 3.5)
project(find_track)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp utils.cpp tracker.cpp tracker.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
