#pragma once
#include <culina/cuda.cuh>
#include <culina/numeric_format/interval.hpp>
#include <culina/types.cuh>
#include <iostream>
#include <memory>

namespace culina_test {
namespace detail {
template <class T> class device_deleter {
public:
  void operator()(T *ptr) { CULINA_CHECK_ERROR(cudaFree(ptr)); }
};
} // namespace detail

template <class T>
using device_unique_ptr = std::unique_ptr<T, detail::device_deleter<T>>;

template <class T>
inline device_unique_ptr<T> get_managed_unique_ptr(const std::size_t count) {
  T *ptr;
  CULINA_CHECK_ERROR_M(cudaMallocManaged(&ptr, sizeof(T) * count),
                       "Failed to allocate " +
                           std::to_string(count * sizeof(T)) +
                           " Bytes of managed memory");
  return device_unique_ptr<T>{ptr};
}

template <class T>
inline device_unique_ptr<T> get_device_unique_ptr(const std::size_t count) {
  T *ptr;
  CULINA_CHECK_ERROR_M(cudaMalloc(&ptr, sizeof(T) * count),
                       "Failed to allocate " +
                           std::to_string(count * sizeof(T)) +
                           " Bytes of device memory");
  return device_unique_ptr<T>{ptr};
}

template <class T> constexpr double get_error_threshold() {
  return get_error_threshold<typename T::data_t>();
}
template <> constexpr double get_error_threshold<double>() { return 1e-14; }
template <> constexpr double get_error_threshold<float>() { return 1e-6; }
template <> constexpr double get_error_threshold<half>() { return 1e-2; }

template <class T> std::string get_str();
template <> std::string get_str<culina::default_mode>() {
  return "default_mode";
}
template <> std::string get_str<culina::generic_kernel_mode>() {
  return "generic_kernel_mode";
}
template <> std::string get_str<double>() { return "double"; }
template <> std::string get_str<float>() { return "float"; }
template <> std::string get_str<half>() { return "half"; }
template <> std::string get_str<culina::numeric_format::interval<double>>() {
  return "interval<double>";
}
template <> std::string get_str<culina::numeric_format::interval<float>>() {
  return "interval<float>";
}
template <> std::string get_str<culina::numeric_format::interval<half>>() {
  return "interval<half>";
}
} // namespace culina_test
