#pragma once
#include <type_traits>
#include "../handle.cuh"
#include "types.cuh"

namespace {
// One y element per warp
template <class Ta, class Tx, class Ty, class Ts, class ComputeT>
__global__ void generic_gemv_kernel(
    const culina::blas::op_t op_a,
    const std::size_t m,
    const std::size_t n,
    const Ts alpha,
    const Ta* a_ptr, const std::size_t lda,
    const Tx* x_ptr, const std::size_t incx,
    const Ts beta,
    Ty* const y_ptr, const std::size_t incy
    ) {
  constexpr std::uint32_t warp_size = 32;
  const auto p = op_a == culina::blas::op_t::N ? m : n;
  const auto k = op_a == culina::blas::op_t::N ? n : m;
  const std::uint64_t pi = (threadIdx.x + blockIdx.x * blockDim.x) / warp_size;
  if (pi >= p) {
    return;
  }

  const auto lane_id = threadIdx.x % warp_size;

  auto c = static_cast<ComputeT>(0);
  if (op_a == culina::blas::op_t::N) {
    for (std::size_t i = lane_id; i < k; i += warp_size) {
      const ComputeT a = a_ptr[i * lda + pi];
      const ComputeT x = x_ptr[i * incx];
      c += a * x;
    }
  } else if (op_a == culina::blas::op_t::T) {
    for (std::size_t i = lane_id; i < k; i += warp_size) {
      const ComputeT a = a_ptr[pi * lda + i];
      const ComputeT x = x_ptr[i * incx];
      c += a * x;
    }
  }
  for(auto mask = (warp_size >> 1); mask > 0; mask >>= 1) {
    c += __shfl_xor_sync(~0u, c, mask);
  }

  if (lane_id == 0) {
    const auto beta_compute = static_cast<ComputeT>(beta);
    const auto alpha_compute = static_cast<ComputeT>(alpha);
    if (beta_compute == static_cast<ComputeT>(0)) {
      y_ptr[pi * incy] = alpha_compute * c;
    } else {
      y_ptr[pi * incy] = alpha_compute * c + beta_compute * y_ptr[pi * incy];
    }
  }
}
} // namespace

namespace culina::blas {
template <class Ta_, class Tx_, class Ty_, class Ts_, class ComputeT_, class Mode_>
struct gemv_policy {
  using Ta = Ta_;
  using Tx = Tx_;
  using Ty = Ty_;
  using Ts = Ts_;
  using ComputeT = ComputeT_;
  using Mode = Mode_;
};

template <class T, class Mode = culina::default_mode>
using default_gemv_polict = gemv_policy<T, T, T, T, T, Mode>;

template <class GemvPolicy>
struct gemv {
  inline culina::status_t operator()(
    culina::handle_base* const handle, 
    const culina::blas::op_t op_a,
    const std::size_t m,
    const std::size_t n,
    const typename GemvPolicy::Ts alpha,
    const typename GemvPolicy::Ta* a_ptr, const std::size_t lda,
    const typename GemvPolicy::Tx* x_ptr, const std::size_t incx,
    const typename GemvPolicy::Ts beta,
    typename GemvPolicy::Ty* const y_ptr, const std::size_t incy
    ) {
    constexpr std::size_t block_size = 256;
    constexpr std::size_t warp_size = 32;
    const std::size_t grid_size = ((op_a == culina::blas::op_t::N ? m : n) * warp_size + block_size - 1) / block_size;

    using Ta = typename GemvPolicy::Ta;
    using Tx = typename GemvPolicy::Tx;
    using Ty = typename GemvPolicy::Ty;
    using Ts = typename GemvPolicy::Ts;
    using ComputeT = typename GemvPolicy::ComputeT;
    using Mode = typename GemvPolicy::Mode;
    generic_gemv_kernel<Ta, Tx, Ty, Ts, ComputeT><<<grid_size, block_size, 0, handle->stream()>>>(
        op_a,
        m, n,
        alpha,
        a_ptr, lda,
        x_ptr, incx,
        beta,
        y_ptr, incy
        );
    return culina::status_t::success;
  }
};

template <>
struct gemv<default_gemv_polict<float, culina::default_mode>> {
  inline culina::status_t operator()(
    culina::handle_base* const handle, 
    const culina::blas::op_t op_a,
    const std::size_t m,
    const std::size_t n,
    const float alpha,
    const float* a_ptr, const std::size_t lda,
    const float* x_ptr, const std::size_t incx,
    const float beta,
    float* const y_ptr, const std::size_t incy
    ) {
    cublasSgemv(
        handle->cublas_handle(),
        culina::blas::to_cublas_op_t(op_a),
        m, n,
        &alpha,
        a_ptr, lda,
        x_ptr, incx,
        &beta,
        y_ptr, incy
        );
    return culina::status_t::success;
  }
};

template <>
struct gemv<default_gemv_polict<double, culina::default_mode>> {
  inline culina::status_t operator()(
    culina::handle_base* const handle, 
    const culina::blas::op_t op_a,
    const std::size_t m,
    const std::size_t n,
    const double alpha,
    const double* a_ptr, const std::size_t lda,
    const double* x_ptr, const std::size_t incx,
    const double beta,
    double* const y_ptr, const std::size_t incy
    ) {
    cublasDgemv(
        handle->cublas_handle(),
        culina::blas::to_cublas_op_t(op_a),
        m, n,
        &alpha,
        a_ptr, lda,
        x_ptr, incx,
        &beta,
        y_ptr, incy
        );
    return culina::status_t::success;
  }
};
} // namespace
