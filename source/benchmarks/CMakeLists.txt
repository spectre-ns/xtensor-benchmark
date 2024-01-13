
cmake_minimum_required(VERSION 3.5)

set(XTENSOR_BENCHMARKS
    reduction_benchmark.cpp
)

# Build
# =====
add_executable(xbenchmark main.cpp ${XTENSOR_BENCHMARKS})
message(STATUS ${XTENSOR_BENCHMARK_INCLUDE_DIR})
target_include_directories(xbenchmark PRIVATE ${XTENSOR_BENCHMARK_INCLUDE_DIR})
target_link_libraries(xbenchmark PRIVATE benchmark::benchmark)