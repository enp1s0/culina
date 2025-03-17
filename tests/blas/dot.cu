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

template <class T, class Mode> void eval(const std::size_t n) {
  const auto incx = 1;
  const auto incy = 2;

  auto vec_x = culina_test::get_managed_unique_ptr<T>(n * incx);
  auto vec_y = culina_test::get_managed_unique_ptr<T>(n * incy);

  std::uniform_real_distribution<double> dist(-1, 1);
  std::mt19937 mt(0);
  for (std::size_t i = 0; i < n; i++) {
    vec_x.get()[i * incx] = static_cast<T>(dist(mt));
  }
  for (std::size_t i = 0; i < n; i++) {
    vec_y.get()[i * incy] = static_cast<T>(dist(mt));
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
  std::printf("%s,%lu,%e,%s\n", typeid(T).name(), n, error,
              (error < error_threshold ? "OK" : "NG"));

  culina_handle->destroy();
  delete culina_handle;
}

int main() {
  std::printf("dtype,m,relative_error,check\n");
  for (std::size_t N = 32; N <= 8192; N <<= 1) {
    eval<half, culina::default_mode>(N);
    eval<float, culina::default_mode>(N);
    eval<double, culina::default_mode>(N);
  }
}
