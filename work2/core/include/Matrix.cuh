#pragma once

#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "Data.cuh"
#include "MatrixView.cuh"
#include "kernel_matmul_naive.cuh"

using namespace std;

// Класс-фасад: объединяет Data (владение памятью) и MatrixView (доступ к данным)
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
    const MatrixView<AtomT>& view() const { return view_; }

    // Умножение: this = lhs * rhs на GPU
    void matmul_from(const Matrix& lhs, const Matrix& rhs);
};

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs);

template <typename AtomT>
void Matrix<AtomT>::matmul_from(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.ncols() != rhs.nrows())
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    if (nrows() != lhs.nrows() || ncols() != rhs.ncols())
        throw invalid_argument("Result matrix has wrong dimensions");

    // Блок 16x16 нитей — стандартный выбор для 2D-задач
    dim3 block(16, 16);
    dim3 grid((ncols() + block.x - 1) / block.x,
              (nrows() + block.y - 1) / block.y);

    kernel_matmul_naive<<<grid, block>>>(lhs.view_, rhs.view_, view_);
    cudaDeviceSynchronize();
}

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
    Matrix<AtomT> result(lhs.nrows(), rhs.ncols());
    result.matmul_from(lhs, rhs);
    return result;
}
