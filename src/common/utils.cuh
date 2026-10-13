#pragma once

#define FULL_MASK 0xffffffff

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
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
