#include <cstddef>
#include <vector>
#include <random>
#include <type_traits>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "Vector.cuh"

using namespace std;

static_assert(is_trivially_copyable_v<VectorView<float>>);

class VectorAddTest : public ::testing::TestWithParam<size_t> {};

TEST_P(VectorAddTest, Addition) {
    size_t n = GetParam();

    vector<float> host_lhs(n);
    vector<float> host_rhs(n);
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < n; ++i) {
        host_lhs[i] = dis(gen);
        host_rhs[i] = dis(gen);
    }

    Eigen::Map<Eigen::VectorXf> eigen_lhs(host_lhs.data(), n);
    Eigen::Map<Eigen::VectorXf> eigen_rhs(host_rhs.data(), n);
    Eigen::VectorXf eigen_result = eigen_lhs + eigen_rhs;

    Vector<float> cuda_lhs(n);
    cuda_lhs.data().copy_from_host(host_lhs.data());

    Vector<float> cuda_rhs(n);
    cuda_rhs.data().copy_from_host(host_rhs.data());

    Vector<float> cuda_result = cuda_lhs + cuda_rhs;

    vector<float> host_result(n);
    cuda_result.data().copy_to_host(host_result.data());

    Eigen::Map<Eigen::VectorXf> custom_eigen_result(host_result.data(), n);
    
    EXPECT_TRUE(eigen_result.isApprox(custom_eigen_result, 1e-6f));
    EXPECT_LE((eigen_result - custom_eigen_result).cwiseAbs().maxCoeff(), 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(
    VectorSizes,
    VectorAddTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029)
);

