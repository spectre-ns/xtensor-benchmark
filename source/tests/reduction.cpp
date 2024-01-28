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
    auto xtensor_result = xt::mean<TypeParam>(input, xt::evaluation_strategy::immediate)();
    auto result = reduction::native::mean(input);
    ASSERT_FLOAT_EQ(result, xtensor_result);
}

TYPED_TEST(ReductionTest, MeanOnSecondAxis)
{
    xt::xarray<TypeParam> input = xt::random::rand<TypeParam>({2,2,2});
    auto xtensor_result = xt::mean(input, 1, xt::evaluation_strategy::immediate);
    auto result = reduction::native::AverageSecondAxis(input);
    std::cout << xtensor_result << std::endl;
    std::cout << result << std::endl;
    ASSERT_TRUE(xt::allclose(result, xtensor_result));
}