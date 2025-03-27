#pragma once
#include "../handle.cuh"
#include "../utils.cuh"
#include "gemm.cuh"
#include "types.cuh"
#include <type_traits>

namespace {
template <class Tx, class Ty, class Tz>
__global__ void
generic_dot_kernel_atomic(Tz *const result_ptr, const Tx *const x_ptr,
                          const std::size_t incx, const Ty *const y_ptr,
                          const std::size_t incy, const std::size_t len) {
  // Check if atomic is supported
  static_assert(std::is_arithmetic_v<Tz> || std::is_same_v<Tz, half>);
  Tz c = 0;
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (std::size_t i = tid; i < len; i += gridDim.x * blockDim.x) {
    c += static_cast<Tz>(x_ptr[i * incx]) * static_cast<Tz>(y_ptr[i * incy]);
  }

  for (std::uint32_t mask = (culina::utils::warp_size >> 1); mask > 0;
       mask >>= 1) {
    c += __shfl_xor_sync(~0u, c, mask);
  }

  if (threadIdx.x % culina::utils::warp_size == 0) {
    culina::utils::atomic_add(result_ptr, c);
  }
}

template <class Tx, class Ty, class Tz, std::uint32_t block_size>
__global__ void
generic_dot_kernel_temp_mem(Tz *const result_ptr, const Tx *const x_ptr,
                            const std::size_t incx, const Ty *const y_ptr,
                            const std::size_t incy, const std::size_t len) {
  Tz c = 0;
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (std::size_t i = tid; i < len; i += gridDim.x * blockDim.x) {
    c = c + static_cast<Tz>(x_ptr[i * incx]) * static_cast<Tz>(y_ptr[i * incy]);
  }

  __shared__ Tz smem[block_size];
  smem[threadIdx.x] = c;

  __syncthreads();
  for (auto mask = (block_size >> 1); mask > 0; mask >>= 1) {
    smem[threadIdx.x] = smem[threadIdx.x] + smem[(threadIdx.x ^ mask)];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result_ptr[blockIdx.x] = smem[0];
  }
}

template <class Tz, std::uint32_t block_size>
__global__ void small_reduction_kernel(Tz *const result_ptr,
                                       const Tz *array_ptr,
                                       const std::uint32_t len) {
  Tz c = 0;

  for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
    c = c + array_ptr[i];
  }

  __shared__ Tz smem[block_size];
  smem[threadIdx.x] = c;

  __syncthreads();
  for (auto mask = (block_size >> 1); mask > 0; mask >>= 1) {
    smem[threadIdx.x] = smem[threadIdx.x] + smem[(threadIdx.x ^ mask)];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result_ptr[blockIdx.x] = smem[0];
  }
}
} // namespace

namespace culina::blas {
template <class Tx_, class Ty_, class Tz_, class Mode_> struct dot_policy {
  using Tx = Tx_;
  using Ty = Ty_;
  using Tz = Tz_;
  using Mode = Mode_;
};

template <class T, class Mode = culina::default_mode>
using default_dot_policy = dot_policy<T, T, T, Mode>;

template <class DotPolicy> struct dot {
  inline culina::status_t
  operator()(culina::handle_base *const handle, const std::size_t n,
             const typename DotPolicy::Tx *const x_ptr, const std::size_t incx,
             const typename DotPolicy::Ty *const y_ptr, const std::size_t incy,
             typename DotPolicy::Tz *const z_ptr) {
    constexpr std::size_t block_size = 256;
    const std::size_t grid_size =
        std::min((n + block_size - 1) / block_size, 1lu << 11);

    using Tx = typename DotPolicy::Tx;
    using Ty = typename DotPolicy::Ty;
    using Tz = typename DotPolicy::Tz;
    using Mode = typename DotPolicy::Mode;

    Tz *tmp_ptr;
    CULINA_CHECK_ERROR(
        cudaMallocAsync(&tmp_ptr, sizeof(Tz) * grid_size, handle->stream()));

    generic_dot_kernel_temp_mem<Tx, Ty, Tz, block_size>
        <<<grid_size, block_size, 0, handle->stream()>>>(tmp_ptr, x_ptr, incx,
                                                         y_ptr, incy, n);
    small_reduction_kernel<Tz, block_size>
        <<<1, block_size, 0, handle->stream()>>>(z_ptr, tmp_ptr, grid_size);

    CULINA_CHECK_ERROR(cudaFreeAsync(tmp_ptr, handle->stream()));

    return culina::status_t::success;
  }
};

template <class Tx, class Ty, class Tz, class Mode>
concept atomic_support = (std::is_arithmetic_v<Tz> || std::is_same_v<Tz, half>);

template <class Tx, class Ty, class Tz, class Mode>
  requires atomic_support<Tx, Ty, Tz, Mode>
struct dot<dot_policy<Tx, Ty, Tz, Mode>> {
  inline culina::status_t operator()(culina::handle_base *const handle,
                                     const std::size_t n, const Tx *const x_ptr,
                                     const std::size_t incx,
                                     const Ty *const y_ptr,
                                     const std::size_t incy, Tz *const z_ptr) {
    constexpr std::size_t block_size = 256;
    const std::size_t grid_size =
        std::min((n + block_size - 1) / block_size, 1lu << 11);

    CULINA_CHECK_ERROR(cudaMemsetAsync(z_ptr, 0, sizeof(Tz), handle->stream()));

    generic_dot_kernel_atomic<Tx, Ty, Tz>
        <<<grid_size, block_size, 0, handle->stream()>>>(z_ptr, x_ptr, incx,
                                                         y_ptr, incy, n);

    return culina::status_t::success;
  }
};

template <> struct dot<default_dot_policy<float>> {
  inline culina::status_t
  operator()(culina::handle_base *const handle, const std::size_t n,
             const float *const x, const std::size_t incx, const float *const y,
             const std::size_t incy, float *const z_ptr) {
    CULINA_CHECK_ERROR(
        cublasSdot(handle->cublas_handle(), n, x, incx, y, incy, z_ptr));

    return culina::status_t::success;
  }
};

template <> struct dot<default_dot_policy<double>> {
  inline culina::status_t operator()(culina::handle_base *const handle,
                                     const std::size_t n, const double *const x,
                                     const std::size_t incx,
                                     const double *const y,
                                     const std::size_t incy,
                                     double *const z_ptr) {
    CULINA_CHECK_ERROR(
        cublasDdot(handle->cublas_handle(), n, x, incx, y, incy, z_ptr));

    return culina::status_t::success;
  }
};
} // namespace culina::blas
