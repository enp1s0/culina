#include "../utils.cuh"
#include <culina/culina.cuh>
#include <random>
#include <typeinfo>

class culina_handle_t : public culina::handle_base {
public:
  virtual ~culina_handle_t() {}

  inline culina::status_t create() {
    culina::handle_base::create();
    return culina::status_t::success;
  }

  inline culina::status_t destroy() {
    culina::handle_base::destroy();
    return culina::status_t::success;
  }
};

template <class T, class Mode> int eval(const std::size_t n) {
  const auto incx = 1;
  const auto incy = 2;

  auto vec_x = culina_test::get_managed_unique_ptr<T>(n * incx);
  auto vec_y = culina_test::get_managed_unique_ptr<T>(n * incy);

  std::uniform_real_distribution<double> dist(-1, 1);
  std::mt19937 mt(0);
  for (std::size_t i = 0; i < n; i++) {
    vec_x.get()[i * incx] = static_cast<T>(dist(mt) * 16);
  }
  for (std::size_t i = 0; i < n; i++) {
    vec_y.get()[i * incy] = static_cast<T>(dist(mt) * 16);
  }

  double c_ref = 0;
#pragma omp parallel for reduction(+ : c_ref)
  for (std::size_t i = 0; i < n; i++) {
    c_ref += static_cast<double>(vec_x.get()[i * incx]) *
             static_cast<double>(vec_y.get()[i * incy]);
  }

  auto *culina_handle = new culina_handle_t;
  culina_handle->create();

  T c;

  using dot_policy = culina::blas::default_dot_policy<T, Mode>;
  culina::blas::dot<dot_policy>{}(culina_handle, n, vec_x.get(), incx,
                                  vec_y.get(), incy, &c);

  CULINA_CHECK_ERROR(cudaStreamSynchronize(culina_handle->stream()));

  double error =
      std::abs((static_cast<double>(c) - c_ref) / static_cast<double>(c_ref));
  const auto error_threshold =
      culina_test::get_error_threshold<T>() * std::sqrt(static_cast<double>(n));
  const auto passed = error < error_threshold;
  if (!passed) {
    std::printf("%s,%lu,%e,%s\n", culina_test::get_str<T>().c_str(), n, error,
                (passed ? "OK" : "NG"));
  }

  culina_handle->destroy();
  delete culina_handle;

  if (!passed) {
    return 1;
  }
  return 0;
}

int main() {
  std::printf("# DOT test\n");
  std::size_t num_tested = 0;
  std::size_t num_failed = 0;
  std::printf("dtype,m,relative_error,check\n");
  for (std::size_t N_base = 32; N_base <= 8192; N_base <<= 1) {
    for (std::size_t N_offset = 0; N_offset < 10; N_offset++) {
      const auto N = N_base + N_offset;
      num_failed += eval<half, culina::default_mode>(N);
      num_failed += eval<float, culina::default_mode>(N);
      num_failed += eval<double, culina::default_mode>(N);
      num_failed +=
          eval<culina::numeric_format::interval<half>, culina::default_mode>(N);
      num_failed +=
          eval<culina::numeric_format::interval<float>, culina::default_mode>(
              N);
      num_failed +=
          eval<culina::numeric_format::interval<double>, culina::default_mode>(
              N);
      num_tested += 6;
    }
  }

  std::printf("Test: %5lu / %5lu passed\n", (num_tested - num_failed),
              num_tested);
}
