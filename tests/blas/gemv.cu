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

template <class T, class Mode>
int eval(const std::size_t m, const std::size_t n,
         const culina::blas::op_t op_A) {
  const auto vec_X_len = op_A == culina::blas::op_t::N ? n : m;
  const auto vec_Y_len = op_A == culina::blas::op_t::N ? m : n;
  const auto lda = m;
  const auto incx = 1;
  const auto incy = 1;
  const auto num_mat_A_elms = lda * n;
  const auto num_vec_X_elms = incx * vec_X_len;
  const auto num_vec_Y_elms = incy * vec_Y_len;

  auto mat_A = culina_test::get_managed_unique_ptr<T>(num_mat_A_elms);
  auto vec_X = culina_test::get_managed_unique_ptr<T>(num_vec_X_elms);
  auto vec_Y = culina_test::get_managed_unique_ptr<T>(num_vec_Y_elms);
  auto vec_Z = culina_test::get_managed_unique_ptr<T>(vec_Y_len);

  std::uniform_real_distribution<double> dist(-1, 1);
  std::mt19937 mt(0);
  for (std::size_t i = 0; i < num_mat_A_elms; i++) {
    mat_A.get()[i] = static_cast<T>(dist(mt));
  }
  for (std::size_t i = 0; i < num_vec_X_elms; i++) {
    vec_X.get()[i] = static_cast<T>(dist(mt));
  }
  for (std::size_t i = 0; i < num_vec_Y_elms; i++) {
    vec_Y.get()[i] = static_cast<T>(dist(mt));
  }

  const T alpha = 1, beta = -1;
#pragma omp parallel for
  for (std::size_t im = 0; im < vec_Y_len; im++) {
    double c = 0;
    for (std::size_t in = 0; in < vec_X_len; in++) {
      const double a =
          mat_A.get()[op_A == culina::blas::op_t::N ? im + in * lda
                                                    : in + im * lda];
      const double x = vec_X.get()[in * incx];
      c += a * x;
    }
    vec_Z.get()[im] =
        c * static_cast<double>(alpha) +
        static_cast<double>(vec_Y.get()[im * incy]) * static_cast<double>(beta);
  }

  auto *culina_handle = new culina_handle_t;
  culina_handle->create();

  using gemv_policy = culina::blas::default_gemv_policy<T, Mode>;
  culina::blas::gemv<gemv_policy>{}(culina_handle, op_A, m, n, alpha,
                                    mat_A.get(), lda, vec_X.get(), incx, beta,
                                    vec_Y.get(), incy);

  CULINA_CHECK_ERROR(cudaStreamSynchronize(culina_handle->stream()));

  double base_norm2 = 0;
  double diff_norm2 = 0;
#pragma omp parallel for reduction(+ : base_norm2) reduction(+ : diff_norm2)
  for (std::size_t im = 0; im < vec_Y_len; im++) {
    const double base = vec_Z.get()[im];
    const double diff = base - static_cast<double>(vec_Y.get()[im * incy]);
    base_norm2 += base * base;
    diff_norm2 += diff * diff;
  }
  const auto error = std::sqrt(diff_norm2 / base_norm2);
  const auto error_threshold =
      culina_test::get_error_threshold<T>() * std::sqrt(static_cast<double>(n));
  const auto passed = error < error_threshold;
  if (!passed) {
    std::printf("%s,%s,%lu,%lu,%e,%s\n", culina_test::get_str<T>().c_str(),
                culina::blas::to_str(op_A).c_str(), m, n, error,
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
  std::printf("# GEMV test\n");
  std::size_t num_tested = 0;
  std::size_t num_failed = 0;
  std::printf("dtype,m,n,relative_error,check\n");
  for (const auto op_A : std::vector<culina::blas::op_t>{
           culina::blas::op_t::N, culina::blas::op_t::T}) {
    for (std::size_t N_base = 32; N_base <= 1024; N_base <<= 1) {
      for (std::size_t N_offset = 0; N_offset < 3; N_offset++) {
        const auto N = N_base + N_offset;
        for (std::size_t M_base = 32; M_base <= 1024; M_base <<= 1) {
          for (std::size_t M_offset = 0; M_offset < 3; M_offset++) {
            const auto M = M_base + M_offset;
            num_failed += eval<half, culina::default_mode>(M, N, op_A);
            num_failed += eval<float, culina::default_mode>(M, N, op_A);
            num_failed += eval<double, culina::default_mode>(M, N, op_A);
            num_failed += eval<culina::numeric_format::interval<half>,
                               culina::default_mode>(M, N, op_A);
            num_failed += eval<culina::numeric_format::interval<float>,
                               culina::default_mode>(M, N, op_A);
            num_failed += eval<culina::numeric_format::interval<double>,
                               culina::default_mode>(M, N, op_A);
            num_tested += 6;
          }
        }
      }
    }
  }
  std::printf("Test: %5lu / %5lu passed\n", (num_tested - num_failed),
              num_tested);
}
