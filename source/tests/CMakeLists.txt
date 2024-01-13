

cmake_minimum_required(VERSION 3.5)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# This will try calling find_package() first for both dependencies
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  xtest
  reduction.cpp
)

target_link_libraries(
  xtest
  gtest_main
)

target_include_directories(xtest PRIVATE ${XTENSOR_BENCHMARK_INCLUDE_DIR})
include(GoogleTest)
gtest_discover_tests(xtest)
