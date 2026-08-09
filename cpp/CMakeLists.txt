cmake_minimum_required(VERSION 3.16)
project(wurld_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(wurld INTERFACE)
target_include_directories(wurld INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(wurld_info wurld_info.cpp)
target_link_libraries(wurld_info PRIVATE wurld)

add_executable(wurld_test test_wurld.cpp)
target_link_libraries(wurld_test PRIVATE wurld)

if(NOT MSVC)
  target_compile_options(wurld_info PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(wurld_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME wurld_unit COMMAND wurld_test)
