#pragma once
#include "../handle.cuh"
#include "types.cuh"
#include <concepts>
#include <type_traits>

namespace {
template <class Ta, class Tb, class Tc, class Ts, class ComputeT,
          class MemIndexT, std::uint32_t block_size>
__global__ void gemm_generic_kernel_no_opt(
    const culina::blas::op_t op_a, const culina::blas::op_t op_b,
    const MemIndexT m, const MemIndexT n, const MemIndexT k, const Ts alpha,
    const Ta *a_ptr, const std::size_t lda, const Tb *b_ptr,
    const std::size_t ldb, const Ts beta, Tc *const c_ptr,
    const std::size_t ldc) {
  constexpr std::uint32_t warp_size = 32;
  const MemIndexT global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
  const std::uint32_t lane_id = threadIdx.x % warp_size;

  if (global_warp_id >= m * n) {
    return;
  }

  const auto mi = global_warp_id % m;
  const auto ni = global_warp_id / m;

  ComputeT sum = 0;
  for (MemIndexT ki = lane_id; ki < k; ki += warp_size) {
    ComputeT a;
    if (op_a == culina::blas::op_t::N) {
      a = a_ptr[ki * lda + mi];
    } else {
      a = a_ptr[ki + mi * lda];
    }

    ComputeT b;
    if (op_b == culina::blas::op_t::N) {
      b = b_ptr[ki + ni * ldb];
    } else {
      b = b_ptr[ki * ldb + ni];
    }

    sum = sum + a * b;
  }

  if constexpr (std::is_arithmetic_v<ComputeT> ||
                std::is_same_v<ComputeT, half>) {
    // If warp shuffle supports ComputeT
    for (auto mask = (warp_size >> 1); mask > 0; mask >>= 1) {
      sum = sum + __shfl_xor_sync(~0u, sum, mask);
    }
  } else {
    __shared__ ComputeT smem[block_size];
    auto local_smem_ptr = smem + (threadIdx.x / warp_size) * warp_size;
    local_smem_ptr[lane_id] = sum;

    __syncwarp();
    for (auto mask = (warp_size >> 1); mask > 0; mask >>= 1) {
      local_smem_ptr[lane_id] =
          local_smem_ptr[lane_id] + local_smem_ptr[(lane_id ^ mask)];
    }

    sum = local_smem_ptr[lane_id];
  }

  if (lane_id == 0) {
    if (beta == static_cast<ComputeT>(0)) {
      c_ptr[mi + ni * ldc] = static_cast<ComputeT>(alpha) * sum;
    } else {
      const ComputeT c = c_ptr[mi + ni * ldc];
      c_ptr[mi + ni * ldc] =
          static_cast<ComputeT>(alpha) * sum + static_cast<ComputeT>(beta) * c;
    }
  }
}
} // namespace

namespace culina::blas {
template <class Ta_, class Tb_, class Tc_, class Ts_, class ComputeT_,
          class Mode_>
struct gemm_policy {
  using Ta = Ta_;
  using Tb = Tb_;
  using Tc = Tc_;
  using Ts = Ts_;
  using ComputeT = ComputeT_;
  using Mode = Mode_;
};

template <class T, class Mode = culina::default_mode>
using default_gemm_policy = gemm_policy<T, T, T, T, T, Mode>;

template <class GemmPolicy> struct gemm {
  inline culina::status_t
  operator()(culina::handle_base *const handle, const culina::blas::op_t op_a,
             const culina::blas::op_t op_b, const std::size_t m,
             const std::size_t n, const std::size_t k,
             const typename GemmPolicy::Ts alpha,
             const typename GemmPolicy::Ta *a_ptr, const std::size_t lda,
             const typename GemmPolicy::Tb *b_ptr, const std::size_t ldb,
             const typename GemmPolicy::Ts beta,
             typename GemmPolicy::Tc *const c_ptr, const std::size_t ldc) {
    constexpr std::size_t block_size = 256;
    constexpr std::size_t warp_size = 32;
    constexpr auto num_warps_per_cta = block_size / warp_size;
    const std::size_t grid_size =
        (m * n + num_warps_per_cta - 1) / num_warps_per_cta;

    using Ta = typename GemmPolicy::Ta;
    using Tb = typename GemmPolicy::Tb;
    using Tc = typename GemmPolicy::Tc;
    using Ts = typename GemmPolicy::Ts;
    using ComputeT = typename GemmPolicy::ComputeT;
    gemm_generic_kernel_no_opt<Ta, Tb, Tc, Ts, ComputeT, std::uint32_t,
                               block_size>
        <<<grid_size, block_size, 0, handle->stream()>>>(
            op_a, op_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr,
            ldc);
    return culina::status_t::success;
  }
};

template <class Ta, class Tb, class Tc, class Ts, class ComputeT, class Mode>
concept cublas_support =
    (((std::is_same_v<Ta, float> || std::is_same_v<Tb, float> ||
       std::is_same_v<Tc, float> || std::is_same_v<Ts, float> ||
       std::is_same_v<ComputeT, float>) &&
      (std::is_same_v<Mode, culina::default_mode> ||
       std::is_same_v<Mode, culina::blas::tensor_op_tf32> ||
       std::is_same_v<Mode, culina::blas::tensor_op_fp16> ||
       std::is_same_v<Mode, culina::blas::tensor_op_bf16>)) ||
     ((std::is_same_v<Ta, double> || std::is_same_v<Tb, double> ||
       std::is_same_v<Tc, double> || std::is_same_v<Ts, double> ||
       std::is_same_v<ComputeT, double>) &&
      (std::is_same_v<Mode, culina::default_mode>)) ||
     ((std::is_same_v<Ta, half> || std::is_same_v<Tb, half> ||
       std::is_same_v<Tc, half> || std::is_same_v<Ts, half> ||
       std::is_same_v<ComputeT, half>) &&
      (std::is_same_v<Mode, culina::default_mode> ||
       std::is_same_v<Mode, culina::blas::tensor_op_fp16>)) ||
     ((std::is_same_v<Ta, half> || std::is_same_v<Tb, half> ||
       std::is_same_v<Tc, half> || std::is_same_v<Ts, half> ||
       std::is_same_v<ComputeT, float>) &&
      (std::is_same_v<Mode, culina::default_mode> ||
       std::is_same_v<Mode, culina::blas::tensor_op_fp16>)));

template <class Ta, class Tb, class Tc, class Ts, class ComputeT, class Mode>
  requires cublas_support<Ta, Tb, Tc, Ts, ComputeT, Mode>
struct gemm<gemm_policy<Ta, Tb, Tc, Ts, ComputeT, Mode>> {
  inline culina::status_t
  operator()(culina::handle_base *const handle, const culina::blas::op_t op_a,
             const culina::blas::op_t op_b, const std::size_t m,
             const std::size_t n, const std::size_t k, const Ts alpha,
             const Ta *a_ptr, const std::size_t lda, const Tb *b_ptr,
             const std::size_t ldb, const Ts beta, Tc *const c_ptr,
             const std::size_t ldc) {
    cublasGemmEx(handle->cublas_handle(), culina::blas::to_cublas_op_t(op_a),
                 culina::blas::to_cublas_op_t(op_b), m, n, k, &alpha, a_ptr,
                 culina::cuda_data_type<Ta>, lda, b_ptr,
                 culina::cuda_data_type<Tb>, ldb, &beta, c_ptr,
                 culina::cuda_data_type<Tc>, ldc,
                 culina::blas::compute_type<ComputeT, Mode>,
                 std::is_same_v<Mode, culina::default_mode>
                     ? CUBLAS_GEMM_DEFAULT
                     : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return culina::status_t::success;
  }
};

} // namespace culina::blas
