cmake_minimum_required(VERSION 3.10)
project(image_format CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(yuv_crop_test main.cpp)
target_link_libraries(yuv_crop_test PRIVATE ${OpenCV_LIBS})
