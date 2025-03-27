#include "../utils.cuh"
#include <chrono>
#include <culina/culina.cuh>
#include <omp.h>
#include <random>

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
int eval(const std::size_t m, const std::size_t n, const std::size_t k,
         const culina::blas::op_t op_A, const culina::blas::op_t op_B) {
  const auto lda = op_A == culina::blas::op_t::N ? m + 20 : k + 31;
  const auto ldb = op_B == culina::blas::op_t::N ? k + 11 : n + 29;
  const auto ldc = m + 10;
  const auto num_mat_A_elms = op_A == culina::blas::op_t::N ? lda * k : m * lda;
  const auto num_mat_B_elms = op_B == culina::blas::op_t::N ? ldb * n : k * ldb;
  const auto num_mat_C_elms = ldc * n;

  auto mat_A = culina_test::get_managed_unique_ptr<T>(num_mat_A_elms);
  auto mat_B = culina_test::get_managed_unique_ptr<T>(num_mat_B_elms);
  auto mat_C = culina_test::get_managed_unique_ptr<T>(num_mat_C_elms);
  auto mat_D = culina_test::get_managed_unique_ptr<T>(num_mat_C_elms);

  std::uniform_real_distribution<double> dist(-1, 1);
#pragma omp parallel
  {
    std::mt19937 mt(omp_get_thread_num());
    for (std::size_t i = omp_get_thread_num(); i < num_mat_A_elms;
         i += omp_get_num_threads()) {
      mat_A.get()[i] = static_cast<T>(dist(mt));
    }
    for (std::size_t i = omp_get_thread_num(); i < num_mat_B_elms;
         i += omp_get_num_threads()) {
      mat_B.get()[i] = static_cast<T>(dist(mt));
    }
    for (std::size_t i = omp_get_thread_num(); i < num_mat_C_elms;
         i += omp_get_num_threads()) {
      mat_C.get()[i] = static_cast<T>(dist(mt));
    }
  }

  const T alpha = 1, beta = -1;
#pragma omp parallel for collapse(2)
  for (std::size_t im = 0; im < m; im++) {
    for (std::size_t in = 0; in < n; in++) {
      double c = 0;
      for (std::size_t ik = 0; ik < k; ik++) {
        const double a =
            mat_A.get()[op_A == culina::blas::op_t::N ? im + ik * lda
                                                      : im * lda + ik];
        const double b =
            mat_B.get()[op_B == culina::blas::op_t::N ? ik + in * ldb
                                                      : ik * ldb + in];
        c += a * b;
      }
      mat_D.get()[im + in * ldc] =
          c * static_cast<double>(alpha) +
          static_cast<double>(mat_C.get()[im + in * ldc]) *
              static_cast<double>(beta);
    }
  }

  auto *culina_handle = new culina_handle_t;
  culina_handle->create();

  CULINA_CHECK_ERROR(cudaStreamSynchronize(culina_handle->stream()));
  const auto start_clock = std::chrono::system_clock::now();

  using gemm_policy = culina::blas::default_gemm_policy<T, Mode>;
  culina::blas::gemm<gemm_policy>{}(culina_handle, op_A, op_B, m, n, k, alpha,
                                    mat_A.get(), lda, mat_B.get(), ldb, beta,
                                    mat_C.get(), ldc);

  CULINA_CHECK_ERROR(cudaStreamSynchronize(culina_handle->stream()));
  const auto end_clock = std::chrono::system_clock::now();
  const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock -
                                                           start_clock)
          .count() *
      1e-9;
  const auto throughput = 2lu * m * n * k / elapsed_time * 1e-9;

  double base_norm2 = 0;
  double diff_norm2 = 0;
#pragma omp parallel for collapse(2) reduction(+ : base_norm2)                 \
    reduction(+ : diff_norm2)
  for (std::size_t im = 0; im < m; im++) {
    for (std::size_t in = 0; in < n; in++) {
      const double base = mat_D.get()[im + in * ldc];
      const double diff =
          base - static_cast<double>(mat_C.get()[im + in * ldc]);
      base_norm2 += base * base;
      diff_norm2 += diff * diff;
    }
  }
  const auto error = std::sqrt(diff_norm2 / base_norm2);
  const auto error_threshold =
      culina_test::get_error_threshold<T>() * std::sqrt(static_cast<double>(k));
  const auto passed = error < error_threshold;
  if (!passed) {
    std::printf(
        "%s,%s,%s,%s,%lu,%lu,%lu,%e,%e,%s\n",
        culina_test::get_str<Mode>().c_str(), culina_test::get_str<T>().c_str(),
        culina::blas::to_str(op_A).c_str(), culina::blas::to_str(op_B).c_str(),
        m, n, k, throughput, error, (error ? "OK" : "NG"));
  }
  std::fflush(stdout);

  culina_handle->destroy();
  delete culina_handle;

  if (!passed) {
    return 1;
  }
  return 0;
}

int main() {
  std::printf("# GEMM test\n");
  std::size_t num_tested = 0;
  std::size_t num_failed = 0;
  std::printf("mode,dtype,m,n,k,gflops,relative_error,check\n");
  for (const auto op_A : std::vector<culina::blas::op_t>{
           culina::blas::op_t::N, culina::blas::op_t::T}) {
    for (const auto op_B : std::vector<culina::blas::op_t>{
             culina::blas::op_t::N, culina::blas::op_t::T}) {
      for (std::size_t N_base = 64; N_base <= 1024; N_base <<= 2) {
        for (std::size_t N_offset = 0; N_offset < 2; N_offset++) {
          const auto N = N_base + N_offset;
          for (std::size_t M_base = 64; M_base <= 1024; M_base <<= 2) {
            for (std::size_t M_offset = 0; M_offset < 2; M_offset++) {
              const auto M = M_base + M_offset;
              for (std::size_t K_base = 64; K_base <= 1024; K_base <<= 2) {
                for (std::size_t K_offset = 0; K_offset < 2; K_offset++) {
                  const auto K = K_base + K_offset;
                  num_failed +=
                      eval<half, culina::default_mode>(M, N, K, op_A, op_B);
                  num_failed +=
                      eval<float, culina::default_mode>(M, N, K, op_A, op_B);
                  num_failed +=
                      eval<double, culina::default_mode>(M, N, K, op_A, op_B);
                  num_failed += eval<half, culina::generic_kernel_mode>(
                      M, N, K, op_A, op_B);
                  num_failed += eval<float, culina::generic_kernel_mode>(
                      M, N, K, op_A, op_B);
                  num_failed += eval<double, culina::generic_kernel_mode>(
                      M, N, K, op_A, op_B);
                  num_failed +=
                      eval<culina::numeric_format::interval<half>,
                           culina::generic_kernel_mode>(M, N, K, op_A, op_B);
                  num_failed +=
                      eval<culina::numeric_format::interval<float>,
                           culina::generic_kernel_mode>(M, N, K, op_A, op_B);
                  num_failed +=
                      eval<culina::numeric_format::interval<double>,
                           culina::generic_kernel_mode>(M, N, K, op_A, op_B);
                  num_tested += 9;
                }
              }
            }
          }
        }
      }
    }
  }
  std::printf("Test: %5lu / %5lu passed\n", (num_tested - num_failed),
              num_tested);
}
