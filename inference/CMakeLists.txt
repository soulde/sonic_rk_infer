cmake_minimum_required(VERSION 3.16)
project(sonic_rk3576_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(RKNNRT_LIBRARY NAMES rknnrt REQUIRED)

add_executable(sonic_rk3576_infer src/main.cpp)
target_include_directories(sonic_rk3576_infer PRIVATE include)
target_link_libraries(sonic_rk3576_infer PRIVATE ${RKNNRT_LIBRARY})
target_compile_options(sonic_rk3576_infer PRIVATE -Wall -Wextra -Wpedantic -O3)

install(TARGETS sonic_rk3576_infer RUNTIME DESTINATION bin)
