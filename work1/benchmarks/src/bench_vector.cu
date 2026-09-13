#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include "Vector.cuh"
#include <vector>
#include <random>

static void BM_EigenVectorAddition(benchmark::State& state) {
    std::size_t n = state.range(0);
    std::vector<float> host_lhs(n);
    std::vector<float> host_rhs(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        host_lhs[i] = dis(gen);
        host_rhs[i] = dis(gen);
    }

    Eigen::Map<Eigen::VectorXf> eigen_lhs(host_lhs.data(), n);
    Eigen::Map<Eigen::VectorXf> eigen_rhs(host_rhs.data(), n);
    Eigen::VectorXf eigen_result(n);

    for (auto _ : state) {
        eigen_result = eigen_lhs + eigen_rhs;
        benchmark::DoNotOptimize(eigen_result);
    }
}

static void BM_CUDAVectorAddition(benchmark::State& state) {
    std::size_t n = state.range(0);
    std::vector<float> host_lhs(n);
    std::vector<float> host_rhs(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        host_lhs[i] = dis(gen);
        host_rhs[i] = dis(gen);
    }

    Vector<float> cuda_lhs(n);
    cuda_lhs.data().copy_from_host(host_lhs.data());

    Vector<float> cuda_rhs(n);
    cuda_rhs.data().copy_from_host(host_rhs.data());
    Vector<float> cuda_result(n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        
        cuda_result.add_from(cuda_lhs, cuda_rhs);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        state.SetIterationTime(milliseconds / 1000.0);
        
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 8, 8^2, 8^3, 8^4, 8^5, 8^6, 8^7, 8^8
BENCHMARK(BM_EigenVectorAddition)
    ->RangeMultiplier(8)
    ->Range(8, 16777216);

BENCHMARK(BM_CUDAVectorAddition)
    ->RangeMultiplier(8)
    ->Range(8, 16777216)
    ->UseManualTime();

