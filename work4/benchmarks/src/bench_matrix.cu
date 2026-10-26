#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include "Matrix.cuh"

using namespace std;

using RowMajorMatrixXh = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Бенчмарк умножения квадратных матриц на CPU (Eigen, half)
static void BM_EigenMatrixMultiplication(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_f(n * n);
    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (float& v : host_f) v = dis(gen);

    RowMajorMatrixXh eigen_a(n, n);
    RowMajorMatrixXh eigen_b(n, n);
    for (int i = 0; i < n * n; ++i) {
        eigen_a.data()[i] = Eigen::half(host_f[i]);
        eigen_b.data()[i] = Eigen::half(host_f[i]);
    }
    RowMajorMatrixXh eigen_result(n, n);

    for (auto _ : state) {
        eigen_result = eigen_a * eigen_b;
        benchmark::DoNotOptimize(eigen_result);
    }
}

// Бенчмарк новой реализации на GPU (CUDA WMMA, тензорные ядра)
static void BM_CUDAMatrixMultiplication_WMMA(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_f(n * n);
    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (float& v : host_f) v = dis(gen);

    vector<__half> host_h(n * n);
    for (int i = 0; i < n * n; ++i) host_h[i] = __float2half(host_f[i]);

    // Выделяем память и копируем данные ДО замеров
    Matrix<__half> cuda_a(n, n);
    cuda_a.data().copy_from_host(host_h.data());

    Matrix<__half> cuda_b(n, n);
    cuda_b.data().copy_from_host(host_h.data());

    Matrix<__half> cuda_result(n, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        cuda_result.matmul_from(cuda_a, cuda_b, MatmulStrategy::WMMA);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Бенчмарк предыдущей версии на GPU (CUDA с разделяемой памятью, __half)
static void BM_CUDAMatrixMultiplication_Shmem(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_f(n * n);
    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (float& v : host_f) v = dis(gen);

    vector<__half> host_h(n * n);
    for (int i = 0; i < n * n; ++i) host_h[i] = __float2half(host_f[i]);

    // Выделяем память и копируем данные ДО замеров
    Matrix<__half> cuda_a(n, n);
    cuda_a.data().copy_from_host(host_h.data());

    Matrix<__half> cuda_b(n, n);
    cuda_b.data().copy_from_host(host_h.data());

    Matrix<__half> cuda_result(n, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        cuda_result.matmul_from(cuda_a, cuda_b, MatmulStrategy::SharedMemory);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Квадратные матрицы: n ∈ {16, 32, 64, 128, 256, 512, 1024}
BENCHMARK(BM_EigenMatrixMultiplication)
    ->RangeMultiplier(2)
    ->Range(16, 1024);

BENCHMARK(BM_CUDAMatrixMultiplication_WMMA)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime();

BENCHMARK(BM_CUDAMatrixMultiplication_Shmem)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime();
