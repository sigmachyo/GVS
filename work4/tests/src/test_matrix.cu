#include <vector>
#include <random>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cuda_fp16.h>
#include "Matrix.cuh"

using namespace std;

// Псевдоним для row-major матрицы Eigen с Eigen::half
using RowMajorMatrixXh = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Параметризованный тест: A(m × k) * B(k × n) = C(m × n)
class MatrixMulTest : public ::testing::TestWithParam<tuple<int, int, int>> {};

TEST_P(MatrixMulTest, Multiplication) {
    auto [m, k, n] = GetParam();

    // Генерируем случайные данные на CPU в float, затем конвертируем в half
    vector<float> host_a_f(m * k);
    vector<float> host_b_f(k * n);

    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (float& v : host_a_f) v = dis(gen);
    for (float& v : host_b_f) v = dis(gen);

    // Конвертируем float -> __half для GPU
    vector<__half> host_a(m * k);
    vector<__half> host_b(k * n);
    for (int i = 0; i < m * k; ++i) host_a[i] = __float2half(host_a_f[i]);
    for (int i = 0; i < k * n; ++i) host_b[i] = __float2half(host_b_f[i]);

    // Эталонный результат через Eigen::half (row-major)
    RowMajorMatrixXh eigen_a(m, k);
    RowMajorMatrixXh eigen_b(k, n);
    for (int i = 0; i < m * k; ++i) eigen_a.data()[i] = Eigen::half(host_a_f[i]);
    for (int i = 0; i < k * n; ++i) eigen_b.data()[i] = Eigen::half(host_b_f[i]);
    RowMajorMatrixXh eigen_result_h = eigen_a * eigen_b;

    // Конвертируем эталон в float для сравнения с isApprox
    RowMajorMatrixXf eigen_result(m, n);
    for (int i = 0; i < m * n; ++i) eigen_result.data()[i] = static_cast<float>(eigen_result_h.data()[i]);

    // Вычисляем на GPU через наш Matrix<__half> (operator* использует kernel_matmul_wmma)
    Matrix<__half> cuda_a(m, k);
    cuda_a.data().copy_from_host(host_a.data());

    Matrix<__half> cuda_b(k, n);
    cuda_b.data().copy_from_host(host_b.data());

    Matrix<__half> cuda_result = cuda_a * cuda_b;

    // Скачиваем результат с GPU
    vector<__half> host_result(m * n);
    cuda_result.data().copy_to_host(host_result.data());

    // Конвертируем __half -> float для сравнения
    RowMajorMatrixXf result_eigen(m, n);
    for (int i = 0; i < m * n; ++i) result_eigen.data()[i] = __half2float(host_result[i]);

    // Сравниваем с эталоном Eigen с точностью 1e-2 (FP16 имеет ограниченную точность)
    EXPECT_TRUE(eigen_result.isApprox(result_eigen, 1e-2f));
}

// 216 комбинаций размеров из задания: m, k, n ∈ {16, 32, 64, 128, 256, 512}
INSTANTIATE_TEST_SUITE_P(
    MatrixSizes,
    MatrixMulTest,
    ::testing::Combine(
        ::testing::Values(16, 32, 64, 128, 256, 512),
        ::testing::Values(16, 32, 64, 128, 256, 512),
        ::testing::Values(16, 32, 64, 128, 256, 512)
    )
);
