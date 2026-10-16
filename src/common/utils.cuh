#pragma once

#include <cstdio>
#include <cstdlib>

#define FULL_MASK 0xffffffff

// When nonzero, CUDA_CHECK exits with status 1 on any CUDA error so callers
// (e.g. the pytest suite) see a failing exit code. Build with
// -DCUDA_CHECK_EXIT_ON_ERROR=0 to only print the error and keep going.
#ifndef CUDA_CHECK_EXIT_ON_ERROR
#define CUDA_CHECK_EXIT_ON_ERROR 1
#endif

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
      if (CUDA_CHECK_EXIT_ON_ERROR) {                                          \
        exit(1);                                                               \
      }                                                                        \
    }                                                                          \
  } while (0)

template <typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
inline __host__ __device__ T align_up(T a, T b) {
  static_assert(std::is_integral_v<T>, "T must be an integer type!");
  return ceil_div(a, b) * b;
}

template <typename T>
inline __host__ __device__ T align_down(T a, T b) {
  static_assert(std::is_integral_v<T>, "T must be an integer type!");
  return a / b * b;
}

inline __device__ unsigned popc_below(const unsigned m,
                                      const unsigned threshold) {
  return __popc(m & ((1u << threshold) - 1u));
}
