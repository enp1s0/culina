#pragma once
#include "../macros.hpp"
#include <cstdint>

namespace culina::numeric_format {
template <class T> struct interval {
  using data_t = T;

  data_t up, down;

  CULINA_DEVICE_HOST interval() {}
  CULINA_DEVICE_HOST interval(const half v)
      : up(static_cast<data_t>(v)), down(static_cast<data_t>(v)) {}
  CULINA_DEVICE_HOST interval(const float v)
      : up(static_cast<data_t>(v)), down(static_cast<data_t>(v)) {}
  CULINA_DEVICE_HOST interval(const double v)
      : up(static_cast<data_t>(v)), down(static_cast<data_t>(v)) {}
  CULINA_DEVICE_HOST interval(const std::int32_t v)
      : up(static_cast<data_t>(v)), down(static_cast<data_t>(v)) {}
  CULINA_DEVICE_HOST interval(const std::int64_t v)
      : up(static_cast<data_t>(v)), down(static_cast<data_t>(v)) {}
  CULINA_DEVICE_HOST interval(const std::uint32_t v)
      : up(static_cast<data_t>(v)), down(static_cast<data_t>(v)) {}
  CULINA_DEVICE_HOST interval(const std::uint64_t v)
      : up(static_cast<data_t>(v)), down(static_cast<data_t>(v)) {}

  CULINA_DEVICE_HOST interval<data_t> operator=(const interval<data_t> &a) {
    up = a.up;
    down = a.down;
    return *this;
  }

  CULINA_DEVICE_HOST bool operator==(const interval<data_t> &a) {
    return a.up == up && a.down == down;
  }

  CULINA_DEVICE_HOST operator double() const {
    return static_cast<double>(up + down) / 2;
  }

  CULINA_DEVICE_HOST operator float() const {
    return static_cast<float>(up + down) / 2;
  }
};

template <class T>
CULINA_DEVICE_HOST interval<T> operator+(const interval<T> &a,
                                         const interval<T> &b) {

  interval<T> c;
  c.up = a.up + b.up;
  c.down = a.down + b.down;

  return c;
}

template <class T>
CULINA_DEVICE_HOST interval<T> operator-(const interval<T> &a,
                                         const interval<T> &b) {

  interval<T> c;
  c.up = a.up - b.down;
  c.down = a.down - b.up;

  return c;
}

template <class T>
CULINA_DEVICE_HOST interval<T> operator*(const interval<T> &a,
                                         const interval<T> &b) {

  interval<T> c;
  const auto c0 = a.up * b.up;
  const auto c1 = a.up * b.down;
  const auto c2 = a.down * b.up;
  const auto c3 = a.down * b.down;

  c.up = max(max(c0, c1), max(c2, c3));
  c.down = min(min(c0, c1), min(c2, c3));

  return c;
}

template <class T>
CULINA_DEVICE_HOST interval<T> operator/(const interval<T> &a,
                                         const interval<T> &b) {

  if (b.up * b.down < static_cast<T>(0)) {
    // Error
  }

  interval<T> b_{.up = static_cast<T>(1) / b.down,
                 .down = static_cast<T>(1) / b.up};

  return a * b_;
}
} // namespace culina::numeric_format
