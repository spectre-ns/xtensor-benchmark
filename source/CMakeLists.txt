
set (CMAKE_CXX_STANDARD 20)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()