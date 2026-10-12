#include <vector>
#include <random>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "Matrix.cuh"

using namespace std;

// Удобный псевдоним для row-major матрицы Eigen (наш порядок хранения)
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Параметризованный тест: A(m x k) * B(k x n) = C(m x n)
class MatrixMulTest : public ::testing::TestWithParam<tuple<int, int, int>> {};

TEST_P(MatrixMulTest, Multiplication) {
    auto [m, k, n] = GetParam();

    // Генерируем случайные данные на CPU
    vector<float> host_a(m * k);
    vector<float> host_b(k * n);

    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (float& v : host_a) v = dis(gen);
    for (float& v : host_b) v = dis(gen);

    // Эталонный результат через Eigen (с row-major маппингом)
    Eigen::Map<RowMajorMatrixXf> eigen_a(host_a.data(), m, k);
    Eigen::Map<RowMajorMatrixXf> eigen_b(host_b.data(), k, n);
    RowMajorMatrixXf eigen_result = eigen_a * eigen_b;

    // Вычисляем на GPU через наш Matrix (operator* использует kernel_matmul_shmem)
    Matrix<float> cuda_a(m, k);
    cuda_a.data().copy_from_host(host_a.data());

    Matrix<float> cuda_b(k, n);
    cuda_b.data().copy_from_host(host_b.data());

    Matrix<float> cuda_result = cuda_a * cuda_b;

    // Скачиваем результат с GPU
    vector<float> host_result(m * n);
    cuda_result.data().copy_to_host(host_result.data());

    // Сравниваем с эталоном Eigen с точностью 1e-5
    Eigen::Map<RowMajorMatrixXf> result_eigen(host_result.data(), m, n);
    EXPECT_TRUE(eigen_result.isApprox(result_eigen, 1e-5f));
}

// 343 комбинации размеров из задания: m, n, k in {1, 2, 3, 127, 128, 129, 512}
INSTANTIATE_TEST_SUITE_P(
    MatrixSizes,
    MatrixMulTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512)
    )
);
