#pragma once
#include "cuda.cuh"
#include <cublas_v2.h>

namespace culina {
enum class status_t { success = 0 };

class handle_base {
protected:
  cublasHandle_t cublas_handle_;
  cudaStream_t cuda_stream_ = 0;

public:
  handle_base() {}
  ~handle_base() {}

  inline cublasHandle_t cublas_handle() const { return cublas_handle_; }

  virtual inline culina::status_t create() {
    CULINA_CHECK_ERROR(cublasCreate(&cublas_handle_));
    CULINA_CHECK_ERROR(cublasSetStream(cublas_handle_, cuda_stream_));
    return culina::status_t::success;
  }

  virtual inline culina::status_t destroy() {
    CULINA_CHECK_ERROR(cublasDestroy(cublas_handle_));
    return culina::status_t::success;
  }

  virtual inline culina::status_t set_stream(cudaStream_t cuda_stream) {
    cuda_stream_ = cuda_stream;
    CULINA_CHECK_ERROR(cublasSetStream(cublas_handle_, cuda_stream_));
    return culina::status_t::success;
  }

  inline cudaStream_t stream() const { return cuda_stream_; }
};
} // namespace culina
