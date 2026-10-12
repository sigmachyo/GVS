#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include "Matrix.cuh"

using namespace std;

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Бенчмарк умножения квадратных матриц на CPU (Eigen)
static void BM_EigenMatrixMultiplication(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_a(n * n);
    vector<float> host_b(n * n);

    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (float& v : host_a) v = dis(gen);
    for (float& v : host_b) v = dis(gen);

    Eigen::Map<RowMajorMatrixXf> eigen_a(host_a.data(), n, n);
    Eigen::Map<RowMajorMatrixXf> eigen_b(host_b.data(), n, n);
    RowMajorMatrixXf eigen_result(n, n);

    for (auto _ : state) {
        eigen_result = eigen_a * eigen_b;
        benchmark::DoNotOptimize(eigen_result);
    }
}

// Бенчмарк новой реализации на GPU (CUDA с разделяемой памятью)
static void BM_CUDAMatrixMultiplication_Shmem(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_a(n * n);
    vector<float> host_b(n * n);

    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (float& v : host_a) v = dis(gen);
    for (float& v : host_b) v = dis(gen);

    // Выделяем память и копируем данные ДО замеров
    Matrix<float> cuda_a(n, n);
    cuda_a.data().copy_from_host(host_a.data());

    Matrix<float> cuda_b(n, n);
    cuda_b.data().copy_from_host(host_b.data());

    Matrix<float> cuda_result(n, n);

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

// Бенчмарк предыдущей версии на GPU (CUDA наивная, без разделяемой памяти)
static void BM_CUDAMatrixMultiplication_Naive(benchmark::State& state) {
    int n = state.range(0);

    vector<float> host_a(n * n);
    vector<float> host_b(n * n);

    mt19937 gen(42);
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (float& v : host_a) v = dis(gen);
    for (float& v : host_b) v = dis(gen);

    // Выделяем память и копируем данные ДО замеров
    Matrix<float> cuda_a(n, n);
    cuda_a.data().copy_from_host(host_a.data());

    Matrix<float> cuda_b(n, n);
    cuda_b.data().copy_from_host(host_b.data());

    Matrix<float> cuda_result(n, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        cuda_result.matmul_from(cuda_a, cuda_b, MatmulStrategy::Naive);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Квадратные матрицы: n in {16, 32, 64, 128, 256, 512, 1024}
BENCHMARK(BM_EigenMatrixMultiplication)
    ->RangeMultiplier(2)
    ->Range(16, 1024);

BENCHMARK(BM_CUDAMatrixMultiplication_Shmem)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime();

BENCHMARK(BM_CUDAMatrixMultiplication_Naive)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime();
