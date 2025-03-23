#pragma once
#include "macros.hpp"

namespace culina::utils {
constexpr std::uint32_t warp_size = 32;

template <class T>
inline CULINA_DEVICE_HOST T atomic_add(T *const ptr, const T v) {
  return atomicAdd(ptr, v);
}

template <>
inline CULINA_DEVICE_HOST half atomic_add<half>(half *const ptr, const half v) {
#if __CUDA_ARCH__ >= 700
  return atomicAdd(ptr, v);
#else
  // TODO
  return static_cast<half>(0);
#endif
}

} // namespace culina::utils

inline CULINA_DEVICE_HOST half max(const half a, const half b) {
  return __hmax(a, b);
}

inline CULINA_DEVICE_HOST half min(const half a, const half b) {
  return __hmin(a, b);
}
