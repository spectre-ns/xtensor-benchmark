#include <gtest/gtest.h>

#include <complex>
#include <xtensor/xarray.hpp>

template <typename T>
class ReductionTest : public testing::Test {};

using MyTypes = ::testing::Types<double, float, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ReductionTest, MyTypes);

TYPED_TEST(ReductionTest, Mean)
{
    
}