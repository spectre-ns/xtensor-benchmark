cmake_minimum_required(VERSION 3.5)
project(xtensor-benchmark)

# Dependencies
# ============

set(xsimd_REQUIRED_VERSION 11.0.0)
if(TARGET xsimd)
    set(xsimd_VERSION ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if(${xsimd_VERSION} VERSION_LESS ${xsimd_REQUIRED_VERSION})
        message(ERROR "Mismatch xsimd versions. Found '${xsimd_VERSION}' but requires: '${xsimd_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xsimd v${xsimd_VERSION}")
    endif()
else()
    find_package(xsimd ${xsimd_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
endif()


set(xtensor_REQUIRED_VERSION 0.23.9)
if(TARGET xtensor)
    set(xtensor_VERSION ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtensor_VERSION} VERSION_GREATER_EQUAL ${xtensor_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()

find_package(benchmark REQUIRED)

add_subdirectory(benchmark)



