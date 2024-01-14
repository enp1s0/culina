#pragma once
#include <type_traits>
#include "../handle.cuh"
#include "types.cuh"

namespace culina::blas {
template <class Ta_, class Tb_, class Tc_, class Ts_, class ComputeT_, class Mode_>
struct gemm_policy {
  using Ta = Ta_;
  using Tb = Tb_;
  using Tc = Tc_;
  using ComputeT = ComputeT_;
  using Mode = Mode_;
};

template <class T, class Mode = culina::default_mode>
using default_gemm_policy = gemm_policy<T, T, T, T, T, Mode>;

template <class GemmPolicy>
struct gemm {
  inline culina::status_t operator()(
    culina::handle_base* const handle, 
    const culina::blas::op_t op_a,
    const culina::blas::op_t op_b,
    const std::size_t m,
    const std::size_t n,
    const std::size_t k,
    const typename GemmPolicy::Ts alpha,
    const typename GemmPolicy::Ta* a_ptr, const std::size_t lda,
    const typename GemmPolicy::Tb* b_ptr, const std::size_t ldb,
    const typename GemmPolicy::Ts beta,
    typename GemmPolicy::Tc* const c_ptr, const std::size_t ldc
    );
};

template <class T, class Mode>
struct gemm<gemm_policy<T, T, T, T, T, Mode>> {
  inline culina::status_t operator()(
    culina::handle_base* const handle, 
    const culina::blas::op_t op_a,
    const culina::blas::op_t op_b,
    const std::size_t m,
    const std::size_t n,
    const std::size_t k,
    const T alpha,
    const T* a_ptr, const std::size_t lda,
    const T* b_ptr, const std::size_t ldb,
    const T beta,
    T* const c_ptr, const std::size_t ldc
    ) {
    cublasGemmEx(
        handle->cublas_handle(),
        culina::blas::to_cublas_op_t(op_a),
        culina::blas::to_cublas_op_t(op_b),
        m, n, k,
        &alpha,
        a_ptr, culina::cuda_data_type<T>, lda,
        b_ptr, culina::cuda_data_type<T>, ldb,
        &beta,
        c_ptr, culina::cuda_data_type<T>, ldc,
        culina::blas::compute_type<T, Mode>,
        std::is_same_v<Mode, culina::default_mode> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    return culina::status_t::success;
  }
};

} // namespace
