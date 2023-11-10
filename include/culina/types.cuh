#pragma once

namespace culina {
class default_mode;

namespace detail {
template <class T>
struct cuda_data_ts;
#define DATA_TYPE_DEF(type_name, number_type, type_size) \
template <> struct cuda_data_ts<type_name> {static constexpr cudaDataType_t value = CUDA_##number_type##_##type_size;}
DATA_TYPE_DEF(half, R, 16F);
DATA_TYPE_DEF(half2, C, 16F);
DATA_TYPE_DEF(__nv_bfloat16, R, 16BF);
DATA_TYPE_DEF(__nv_bfloat162, C, 16BF);
DATA_TYPE_DEF(float, R, 32F);
DATA_TYPE_DEF(cuComplex, C, 32F);
DATA_TYPE_DEF(double, R, 64F);
DATA_TYPE_DEF(cuDoubleComplex, C, 64F);
}
template <class T>
constexpr cudaDataType_t cuda_data_type = detail::cuda_data_ts<T>::value;
} // namespace culina
