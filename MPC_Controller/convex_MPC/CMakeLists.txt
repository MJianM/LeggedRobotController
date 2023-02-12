project(MPC CXX)

cmake_minimum_required(VERSION 3.4...3.18)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# 把当前源文件路径添加到CMAKE_MUDULE搜索路径
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# 搜索eigen3文件路径
find_package(osqp REQUIRED)
find_package(pybind11 REQUIRED)


# 把eigen3相关添加到include路径
include_directories("${PROJECT_SOURCE_DIR}/../../extern/qpoases/include" "/usr/local/include")
# 设置寻找库文件的路径
link_directories(APPEND ${LINK_DIRECTORIES} "/usr/local/lib" "${SOURCE_DIR}/../../extern/qpoases/bin")

# set(PYBIND_DIR ${CMAKE_SOURCE_DIR}/../../extern)
# add_subdirectory(${PYBIND_DIR}/pybind11 pybind11.out)
pybind11_add_module(mpc_osqp mpc_osqp.cc)

# add_library(mpc_osqp mpc_osqp.cc)
target_link_libraries(mpc_osqp PUBLIC osqp::osqpstatic PUBLIC /home/mjm/MPC/MPC/extern/qpoases/bin/libqpOASES.so)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off