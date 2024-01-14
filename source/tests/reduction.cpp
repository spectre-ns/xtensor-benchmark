#include "reduction.hpp"

#include <gtest/gtest.h>
#include <complex>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

template <typename T>
class ReductionTest : public testing::Test {};

using MyTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(ReductionTest, MyTypes);

TYPED_TEST(ReductionTest, Mean)
{
    xt::xarray<TypeParam> input = xt::random::rand<TypeParam>({10,10,10});
    auto result = reduction::native::mean(input);
    auto xtensor_result = xt::mean(input)();
    ASSERT_FLOAT_EQ(result, xtensor_result);
}