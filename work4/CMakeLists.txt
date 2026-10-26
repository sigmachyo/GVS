cmake_minimum_required(VERSION 3.20)
project(work4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_LESS "3.25")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
else()
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# WMMA требует архитектуры sm_70+ (Volta и новее)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)
FetchContent_MakeAvailable(eigen)

add_subdirectory(core)
enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
