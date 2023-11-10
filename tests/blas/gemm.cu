#include <random>
#include <typeinfo>
#include <culina/culina.cuh>
#include "../utils.cuh"

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
void eval(
    const std::size_t m,
    const std::size_t n,
    const std::size_t k,
    const culina::blas::op_t op_A,
    const culina::blas::op_t op_B
    ) {
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
  std::mt19937 mt(0);
  for (std::size_t i = 0; i < num_mat_A_elms; i++) {mat_A.get()[i] = static_cast<T>(dist(mt));}
  for (std::size_t i = 0; i < num_mat_B_elms; i++) {mat_B.get()[i] = static_cast<T>(dist(mt));}
  for (std::size_t i = 0; i < num_mat_C_elms; i++) {mat_C.get()[i] = static_cast<T>(dist(mt));}

  const T alpha = 1, beta = -1;
#pragma omp parallel for collapse(2)
  for (std::size_t im = 0; im < m; im++) {
    for (std::size_t in = 0; in < n; in++) {
      double c = 0;
      for (std::size_t ik = 0; ik < k; ik++) {
        const double a = mat_A.get()[op_A == culina::blas::op_t::N ? im + ik * lda : im * lda + ik];
        const double b = mat_B.get()[op_B == culina::blas::op_t::N ? ik + in * ldb : ik * ldb + in];
        c += a * b;
      }
      mat_D.get()[im + in * ldc] = c * static_cast<double>(alpha) + static_cast<double>(mat_C.get()[im + in * ldc]) * static_cast<double>(beta);
    }
  }

  auto* culina_handle = new culina_handle_t;
  culina_handle->create();

  culina::blas::gemm<T, T, T, T, Mode>{}(
      culina_handle,
      op_A,
      op_B,
      m, n, k,
      alpha,
      mat_A.get(), lda,
      mat_B.get(), ldb,
      beta,
      mat_C.get(), ldc
      );

  CULINA_CHECK_ERROR(cudaStreamSynchronize(culina_handle->stream()));

  double base_norm2 = 0;
  double diff_norm2 = 0;
#pragma omp parallel for collapse(2) reduction(+: base_norm2) reduction(+: diff_norm2)
  for (std::size_t im = 0; im < m; im++) {
    for (std::size_t in = 0; in < n; in++) {
      const double base = mat_D.get()[im + in * ldc];
      const double diff = base - static_cast<double>(mat_C.get()[im + in * ldc]);
      base_norm2 += base * base;
      diff_norm2 += diff * diff;
    }
  }
  const auto error = std::sqrt(diff_norm2 / base_norm2);
  const auto error_threshold = culina_test::get_error_threshold<T>() * std::sqrt(static_cast<double>(k));
  std::printf("%s,%s,%s,%lu,%lu,%lu,%e,%s\n",
              typeid(T).name(),
              culina::blas::to_str(op_A).c_str(),
              culina::blas::to_str(op_B).c_str(),
              m, n, k,
              error,
              (error < error_threshold ? "OK" : "NG")
              );

  culina_handle->destroy();
  delete culina_handle;
}

int main() {
  std::printf("dtype,m,n,k,relative_error,check\n");
  for (const auto op_A : std::vector<culina::blas::op_t>{culina::blas::op_t::N, culina::blas::op_t::T}) {
    for (const auto op_B : std::vector<culina::blas::op_t>{culina::blas::op_t::N, culina::blas::op_t::T}) {
      eval<half  , culina::default_mode>(1024, 512, 256, op_A, op_B);
      eval<float , culina::default_mode>(1024, 512, 256, op_A, op_B);
      eval<double, culina::default_mode>(1024, 512, 256, op_A, op_B);
    }
  }
}
