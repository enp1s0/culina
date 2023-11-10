#pragma once
#include <iostream>
#include <sstream>
#include <cassert>

#define CULINA_CHECK_ERROR(status)            culina::detail::check_error(status, __FILE__, __LINE__, __func__)
#define CULINA_CHECK_ERROR_M(status, message) culina::detail::check_error(status, __FILE__, __LINE__, __func__, (message))

namespace culina::detail {
inline void check_error(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
  if(error != cudaSuccess){
    std::stringstream ss;
    ss << cudaGetErrorString( error );
    if(message.length() != 0){
      ss << " : " << message;
    }
    ss << " [" << filename << ":" << line << " in " << funcname << "]";
    throw std::runtime_error(ss.str());
  }
}

inline void check_error(cublasStatus_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
  if(error != CUBLAS_STATUS_SUCCESS){
    std::string error_string;
#define CUBLAS_ERROR_CASE(c) case c: error_string = #c; break
    switch(error){
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_SUCCESS );
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_NOT_INITIALIZED );
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_ALLOC_FAILED );
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_INVALID_VALUE );
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_ARCH_MISMATCH );
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_MAPPING_ERROR );
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_EXECUTION_FAILED );
      CUBLAS_ERROR_CASE( CUBLAS_STATUS_INTERNAL_ERROR );
      default: error_string = "Unknown error"; break;
    }
    std::stringstream ss;
    ss<< error_string;
    if(message.length() != 0){
      ss << " : " << message;
    }
    ss << " [" << filename << ":" << line << " in " << funcname << "]";
    throw std::runtime_error(ss.str());
  }
}
} // namespace culina::detail
