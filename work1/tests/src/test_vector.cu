#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "Vector.cuh"
#include <vector>
#include <random>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<VectorView<float>>);

class VectorAddTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorAddTest, Addition) {
    std::size_t n = GetParam();

    // Generate random data
    std::vector<float> host_lhs(n);
    std::vector<float> host_rhs(n);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        host_lhs[i] = dis(gen);
        host_rhs[i] = dis(gen);
    }

    // Initialize Eigen vectors
    Eigen::Map<Eigen::VectorXf> eigen_lhs(host_lhs.data(), n);
    Eigen::Map<Eigen::VectorXf> eigen_rhs(host_rhs.data(), n);
    Eigen::VectorXf eigen_result = eigen_lhs + eigen_rhs;

    // Initialize custom CUDA vectors
    Vector<float> cuda_lhs(n);
    cuda_lhs.data().copy_from_host(host_lhs.data());

    Vector<float> cuda_rhs(n);
    cuda_rhs.data().copy_from_host(host_rhs.data());

    // Perform CUDA addition
    Vector<float> cuda_result = cuda_lhs + cuda_rhs;

    // Copy result back to host
    std::vector<float> host_result(n);
    cuda_result.data().copy_to_host(host_result.data());

    // Compare with Eigen
    Eigen::Map<Eigen::VectorXf> custom_eigen_result(host_result.data(), n);
    
    // isApprox is required by the assignment; the second check enforces absolute error.
    EXPECT_TRUE(eigen_result.isApprox(custom_eigen_result, 1e-6f));
    EXPECT_LE((eigen_result - custom_eigen_result).cwiseAbs().maxCoeff(), 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(
    VectorSizes,
    VectorAddTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029)
);

