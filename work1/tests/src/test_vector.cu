#include <vector>
#include <random>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "Vector.cuh"

using namespace std;

// Параметризованный тест для разных размеров вектора
class VectorAddTest : public ::testing::TestWithParam<int> {};

TEST_P(VectorAddTest, Addition) {
    int n = GetParam();

    // Генерируем случайные данные на CPU
    vector<float> host_lhs(n);
    vector<float> host_rhs(n);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        host_lhs[i] = dis(gen);
        host_rhs[i] = dis(gen);
    }

    // Эталонный результат через Eigen (CPU)
    Eigen::Map<Eigen::VectorXf> eigen_lhs(host_lhs.data(), n);
    Eigen::Map<Eigen::VectorXf> eigen_rhs(host_rhs.data(), n);
    Eigen::VectorXf eigen_result = eigen_lhs + eigen_rhs;

    // Вычисляем на GPU через наш Vector
    Vector<float> cuda_lhs(n);
    cuda_lhs.data().copy_from_host(host_lhs.data());

    Vector<float> cuda_rhs(n);
    cuda_rhs.data().copy_from_host(host_rhs.data());

    Vector<float> cuda_result = cuda_lhs + cuda_rhs;

    // Скачиваем результат с GPU
    vector<float> host_result(n);
    cuda_result.data().copy_to_host(host_result.data());

    // Сравниваем с эталоном Eigen
    Eigen::Map<Eigen::VectorXf> result_eigen(host_result.data(), n);
    EXPECT_TRUE(eigen_result.isApprox(result_eigen, 1e-6f));
}

// Размеры из задания
INSTANTIATE_TEST_SUITE_P(
    VectorSizes,
    VectorAddTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029)
);
