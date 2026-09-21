#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include "Vector.cuh"

using namespace std;

// Бенчмарк сложения на CPU (Eigen)
static void BM_EigenVectorAddition(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_lhs(n);
    vector<float> host_rhs(n);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
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

// Бенчмарк сложения на GPU (CUDA) с использованием CUDA Events для замера времени
static void BM_CUDAVectorAddition(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_lhs(n);
    vector<float> host_rhs(n);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        host_lhs[i] = dis(gen);
        host_rhs[i] = dis(gen);
    }

    // Выделяем память и копируем данные до замеров
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

// Размеры: 8^1, 8^2, ..., 8^8
BENCHMARK(BM_EigenVectorAddition)
    ->RangeMultiplier(8)
    ->Range(8, 16777216);

BENCHMARK(BM_CUDAVectorAddition)
    ->RangeMultiplier(8)
    ->Range(8, 16777216)
    ->UseManualTime();
