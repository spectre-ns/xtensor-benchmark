

if(XTENSOR_BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
endif()

if(XTENSOR_BUILD_TESTS)
    add_subdirectory(tests)
endif()