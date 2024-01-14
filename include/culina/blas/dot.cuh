#pragma once
#include <type_traits>
#include "../handle.cuh"
#include "../utils.cuh"
#include "types.cuh"
#include "gemm.cuh"

namespace {
template <class Tx, class Ty, class Tz>
__global__ void generic_dot_kernel(
    Tz* const result_ptr,
    const Tx* const x_ptr, const std::size_t incx,
    const Ty* const y_ptr, const std::size_t incy,
    const std::size_t len
    ) {
  Tz c = 0;
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (std::size_t i = tid; i < len; i += gridDim.x * blockDim.x) {
    c += static_cast<Tz>(x_ptr[i * incx]) * static_cast<Tz>(y_ptr[i * incy]);
  }

  for(std::uint32_t mask = (culina::utils::warp_size >> 1); mask > 0; mask >>= 1) {
    c += __shfl_xor_sync(~0u, c, mask);
  }

  if (threadIdx.x % culina::utils::warp_size == 0) {
    culina::utils::atomic_add(result_ptr, c);
  }
}
} // namespace

namespace culina::blas {
template <class Tx_, class Ty_, class Tz_, class Mode_>
struct dot_policy {
  using Tx = Tx_;
  using Ty = Ty_;
  using Tz = Tz_;
  using Mode = Mode_;
};

template <class T, class Mode = culina::default_mode>
using default_dot_policy = dot_policy<T, T, T, Mode>;

template <class DotPolicy>
struct dot {
  inline culina::status_t operator()(
      culina::handle_base* const handle, 
      const std::size_t n,
      const typename DotPolicy::Tx* const x_ptr, const std::size_t incx,
      const typename DotPolicy::Ty* const y_ptr, const std::size_t incy,
      typename DotPolicy::Tz* const z_ptr
      ) {
    constexpr std::size_t block_size = 256;
    const std::size_t grid_size = (n + block_size - 1) / block_size;

    using Tx = typename DotPolicy::Tx;
    using Ty = typename DotPolicy::Ty;
    using Tz = typename DotPolicy::Tz;
    using Mode = typename DotPolicy::Mode;

    Tz* c_ptr;
    CULINA_CHECK_ERROR(cudaMallocAsync(&c_ptr, sizeof(Tz), handle->stream()));
    CULINA_CHECK_ERROR(cudaMemsetAsync(c_ptr, 0, sizeof(Tz), handle->stream()));

    generic_dot_kernel<Tx, Ty, Tz><<<grid_size, block_size, 0, handle->stream()>>>(
        c_ptr,
        x_ptr, incx,
        y_ptr, incy,
        n
        );

    CULINA_CHECK_ERROR(cudaMemcpyAsync(z_ptr, c_ptr, sizeof(Tz), cudaMemcpyDefault, handle->stream()));
    CULINA_CHECK_ERROR(cudaFreeAsync(c_ptr, handle->stream()));

    return culina::status_t::success;
  }
};

template <>
struct dot<default_dot_policy<float>> {
  inline culina::status_t operator()(
      culina::handle_base* const handle, 
      const std::size_t n,
      const float* const x, const std::size_t incx,
      const float* const y, const std::size_t incy,
      float* const z_ptr
      ) {
    CULINA_CHECK_ERROR(cublasSdot(handle->cublas_handle(), n, x, incx, y, incy, z_ptr));

    return culina::status_t::success;
  }
};

template <>
struct dot<default_dot_policy<double>> {
  inline culina::status_t operator()(
      culina::handle_base* const handle, 
      const std::size_t n,
      const double* const x, const std::size_t incx,
      const double* const y, const std::size_t incy,
      double* const z_ptr
      ) {
    CULINA_CHECK_ERROR(cublasDdot(handle->cublas_handle(), n, x, incx, y, incy, z_ptr));

    return culina::status_t::success;
  }
};
} // namespace culina::blas
