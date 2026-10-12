#pragma once

#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "Data.cuh"
#include "MatrixView.cuh"
#include "kernel_matmul_naive.cuh"
#include "kernel_matmul_shmem.cuh"

using namespace std;

// Паттерн «Стратегия» для выбора алгоритма умножения матриц
enum class MatmulStrategy {
    Naive,        // Наивный алгоритм (из работы №2)
    SharedMemory  // Блочный алгоритм с разделяемой памятью (работа №3)
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
                     MatmulStrategy strategy = MatmulStrategy::SharedMemory);
};

// Перегрузка оператора умножения: по умолчанию использует разделяемую память (work3)
template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs);

template <typename AtomT>
void Matrix<AtomT>::matmul_from(const Matrix& lhs, const Matrix& rhs, MatmulStrategy strategy) {
    if (lhs.ncols() != rhs.nrows())
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    if (nrows() != lhs.nrows() || ncols() != rhs.ncols())
        throw invalid_argument("Result matrix has wrong dimensions");

    // Размер блока 16x16 нитей
    constexpr int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((ncols() + block.x - 1) / block.x,
              (nrows() + block.y - 1) / block.y);

    if (strategy == MatmulStrategy::SharedMemory) {
        kernel_matmul_shmem<AtomT, BLOCK_SIZE><<<grid, block>>>(lhs.view(), rhs.view(), view_);
    } else {
        kernel_matmul_naive<AtomT><<<grid, block>>>(lhs.view(), rhs.view(), view_);
    }
    cudaDeviceSynchronize();
}

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
    Matrix<AtomT> result(lhs.nrows(), rhs.ncols());
    result.matmul_from(lhs, rhs, MatmulStrategy::SharedMemory);
    return result;
}
