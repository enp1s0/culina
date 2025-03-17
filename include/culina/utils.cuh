#pragma once
namespace culina::utils {
constexpr std::uint32_t warp_size = 32;

template <class T> inline __device__ T atomic_add(T *const ptr, const T v) {
  return atomicAdd(ptr, v);
}

template <>
inline __device__ half atomic_add<half>(half *const ptr, const half v) {
#if __CUDA_ARCH__ >= 700
  return atomicAdd(ptr, v);
#else
  // TODO
#endif
}
} // namespace culina::utils
