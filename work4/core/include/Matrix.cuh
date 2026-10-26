#pragma once

#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "Data.cuh"
#include "MatrixView.cuh"
#include "kernel_matmul_naive.cuh"
#include "kernel_matmul_shmem.cuh"
#include "kernel_matmul_wmma.cuh"

using namespace std;

// Паттерн «Стратегия» для выбора алгоритма умножения матриц
enum class MatmulStrategy {
    Naive,        // Наивный алгоритм (из работы №2)
    SharedMemory, // Блочный алгоритм с разделяемой памятью (работа №3)
    WMMA          // Тензорные ядра CUDA WMMA (работа №4)
};

// Класс-фасад: объединяет Data (владение памятью GPU) и MatrixView (доступ к данным)
template <typename AtomT>
class Matrix {
private:
    shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;

public:
    Matrix(int nrows, int ncols)
        : data_(make_shared<Data<AtomT>>(nrows * ncols)),
          view_(data_->data(), nrows, ncols) {}

    int nrows() const { return view_.nrows(); }
    int ncols() const { return view_.ncols(); }
    int size()  const { return view_.size(); }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    MatrixView<AtomT>& view() { return view_; }
    MatrixView<const AtomT> view() const { return view_; }
    MatrixView<const AtomT> cview() const { return view_; }

    // Умножение: this = lhs * rhs на GPU с поддержкой паттерна Стратегия
    void matmul_from(const Matrix& lhs, const Matrix& rhs,
                     MatmulStrategy strategy = MatmulStrategy::WMMA);
};

// Перегрузка оператора умножения: по умолчанию использует WMMA (work4)
template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs);

template <typename AtomT>
void Matrix<AtomT>::matmul_from(const Matrix& lhs, const Matrix& rhs, MatmulStrategy strategy) {
    if (lhs.ncols() != rhs.nrows())
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    if (nrows() != lhs.nrows() || ncols() != rhs.ncols())
        throw invalid_argument("Result matrix has wrong dimensions");

    if (strategy == MatmulStrategy::WMMA) {
        // WMMA: каждый варп (32 нити) вычисляет один тайл 16×16 результата
        // Блок 128×4 = 512 нитей = 16 варпов, покрывает 4×4 WMMA-тайлов (64×64 элементов)
        constexpr int WARP_SIZE = 32;
        dim3 block(128, 4);
        // Число варпов по X и Y в одном блоке
        int warpsPerBlockX = block.x / WARP_SIZE;  // 4
        int warpsPerBlockY = block.y;               // 4
        dim3 grid((ncols() + warpsPerBlockX * WMMA_N - 1) / (warpsPerBlockX * WMMA_N),
                  (nrows() + warpsPerBlockY * WMMA_M - 1) / (warpsPerBlockY * WMMA_M));

        kernel_matmul_wmma<<<grid, block>>>(lhs.view(), rhs.view(), view_);
    } else if (strategy == MatmulStrategy::SharedMemory) {
        constexpr int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((ncols() + block.x - 1) / block.x,
                  (nrows() + block.y - 1) / block.y);
        kernel_matmul_shmem<AtomT, BLOCK_SIZE><<<grid, block>>>(lhs.view(), rhs.view(), view_);
    } else {
        constexpr int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((ncols() + block.x - 1) / block.x,
                  (nrows() + block.y - 1) / block.y);
        kernel_matmul_naive<AtomT><<<grid, block>>>(lhs.view(), rhs.view(), view_);
    }
    cudaDeviceSynchronize();
}

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
    Matrix<AtomT> result(lhs.nrows(), rhs.ncols());
    result.matmul_from(lhs, rhs, MatmulStrategy::WMMA);
    return result;
}
