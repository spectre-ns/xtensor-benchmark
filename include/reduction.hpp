

#include <xtensor/xarray.hpp>

#include <numeric>
#include <algorithm>

namespace reduction 
{
    namespace native
    {
        template<class T>
        auto AverageSecondAxis(const xt::xarray<T>& input)
        {
            xt::xarray<T> output = xt::xarray<T>::from_shape({input.shape(0), input.shape(2)});
            const size_t firstAxis = input.shape(0);
            const size_t secondAxis = input.shape(1);
            const size_t lastAxis = input.shape(2);
            for (auto i = 0; i < firstAxis; ++i)  // line group
            {
                // reverse the logical order to get coalesced memory access
                for (size_t k = 0; k < secondAxis; k++)  // within group
                {
                    auto subsetOut = &(output.data()[i * lastAxis]);
                    const auto subsetIn = &(input.data()[(k + i * secondAxis) * lastAxis]);
                    for (auto j = 0; j < lastAxis; j++)  // depth
                    {
                        subsetOut[j] += subsetIn[j];
                    }
                }
            }
            std::ranges::transform(output.begin(), output.end(), output.begin(), [&](auto value) { return value / secondAxis; });
            return output;
        }
    
        template<class T>
        auto Mean(const xt::xarray<T>& input)
        {
            xt::xarray<T> output = xt::xarray<T>::from_shape(input.shape());
            auto result = std::reduce(input.begin(), input.end(), 0) / input.size();
            return result;
        }
    }
}

