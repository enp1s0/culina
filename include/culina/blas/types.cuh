#pragma once
#include "../types.cuh"
#include <cublas_v2.h>
#include <string>

namespace culina::blas {
enum class op_t { N, T, C };

cublasOperation_t to_cublas_op_t(const culina::blas::op_t op) {
  if (op == culina::blas::op_t::N) {
    return CUBLAS_OP_N;
  } else if (op == culina::blas::op_t::T) {
    return CUBLAS_OP_T;
  }
  return CUBLAS_OP_C;
}

std::string to_str(const culina::blas::op_t op) {
  if (op == culina::blas::op_t::N) {
    return "N";
  } else if (op == culina::blas::op_t::T) {
    return "T";
  }
  return "C";
}

// Using Tensor Cores
class tensor_op_tf32;
class tensor_op_fp16;
class tensor_op_bf16;

namespace detail {
template <class T, class Mode> struct cuda_type_ts;
template <> struct cuda_type_ts<double, culina::default_mode> {
  constexpr static cublasComputeType_t value = CUBLAS_COMPUTE_64F;
};
template <> struct cuda_type_ts<float, culina::default_mode> {
  constexpr static cublasComputeType_t value = CUBLAS_COMPUTE_32F;
};
template <> struct cuda_type_ts<half, culina::default_mode> {
  constexpr static cublasComputeType_t value = CUBLAS_COMPUTE_16F;
};
template <> struct cuda_type_ts<float, culina::blas::tensor_op_tf32> {
  constexpr static cublasComputeType_t value = CUBLAS_COMPUTE_32F_FAST_TF32;
};
template <> struct cuda_type_ts<float, culina::blas::tensor_op_fp16> {
  constexpr static cublasComputeType_t value = CUBLAS_COMPUTE_32F_FAST_16F;
};
template <> struct cuda_type_ts<float, culina::blas::tensor_op_bf16> {
  constexpr static cublasComputeType_t value = CUBLAS_COMPUTE_32F_FAST_16BF;
};
} // namespace detail
template <class T, class Mode>
constexpr cublasComputeType_t compute_type =
    detail::cuda_type_ts<T, Mode>::value;
} // namespace culina::blas
