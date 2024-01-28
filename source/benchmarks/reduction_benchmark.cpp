#include "reduction.hpp"

#include <xtensor/xrandom.hpp>
#include <benchmark/benchmark.h>

template<class T>
static void native_mean(benchmark::State& state) {
  xt::xarray<T> input = xt::random::rand<T>({state.range(0),state.range(0),state.range(0)});
  for (auto _ : state) {
        benchmark::DoNotOptimize(reduction::native::mean(input));
  }
}

template<class T>
static void xtensor_mean(benchmark::State& state) {
  xt::xarray<T> input = xt::random::rand<T>({state.range(0),state.range(0),state.range(0)});
  for (auto _ : state) {
    benchmark::DoNotOptimize(xt::mean(input, xt::evaluation_strategy::immediate)());
  }
}

template<class T>
static void native_mean_on_second_axis(benchmark::State& state) {
  xt::xarray<T> input = xt::random::rand<T>({state.range(0), state.range(0), state.range(0)});
  for (auto _ : state) {
    benchmark::DoNotOptimize(reduction::native::AverageSecondAxis(input));
  }
}

template<class T>
static void xtensor_mean_on_second_axis(benchmark::State& state) {
  xt::xarray<T> input = xt::random::rand<T>({state.range(0), state.range(0), state.range(0)});
  for (auto _ : state) {
    benchmark::DoNotOptimize(xt::mean(input, 1, xt::evaluation_strategy::immediate)());
  }
}

// Register the function as a benchmark
BENCHMARK_TEMPLATE(native_mean, float)->Arg(8)->Arg(64);
BENCHMARK_TEMPLATE(native_mean, double)->Arg(8)->Arg(64);
BENCHMARK_TEMPLATE(xtensor_mean, float)->Arg(8)->Arg(64);
BENCHMARK_TEMPLATE(xtensor_mean, double)->Arg(8)->Arg(64);
BENCHMARK_TEMPLATE(xtensor_mean_on_second_axis, float)->Arg(8)->Arg(64);
BENCHMARK_TEMPLATE(xtensor_mean_on_second_axis, double)->Arg(8)->Arg(64);
BENCHMARK_TEMPLATE(native_mean_on_second_axis, float)->Arg(8)->Arg(64);
BENCHMARK_TEMPLATE(native_mean_on_second_axis, double)->Arg(8)->Arg(64);