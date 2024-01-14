#include "reduction.hpp"

#include <xtensor/xrandom.hpp>
#include <benchmark/benchmark.h>

template<class T>
static void native_mean(benchmark::State& state) {
  xt::xarray<T> input = xt::random::rand<T>({state.range(0),state.range(0),state.range(0)});
  for (auto _ : state) {
    benchmark::DoNotOptimize(xt::mean(input));
  }
}

template<class T>
static void xtensor_mean(benchmark::State& state) {
  xt::xarray<T> input = xt::random::rand<T>({state.range(0),state.range(0),state.range(0)});
  for (auto _ : state) {
    benchmark::DoNotOptimize(reduction::native::mean(input));
  }
}

// Register the function as a benchmark
BENCHMARK_TEMPLATE(native_mean, float)->RangeMultiplier(2)->Range(2, 2<<8);
BENCHMARK_TEMPLATE(native_mean, double)->RangeMultiplier(2)->Range(2, 2<<8);
BENCHMARK_TEMPLATE(xtensor_mean, float)->RangeMultiplier(2)->Range(2, 2<<8);
BENCHMARK_TEMPLATE(xtensor_mean, double)->RangeMultiplier(2)->Range(2, 2<<8);