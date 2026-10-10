#pragma once

#include <cuda_runtime.h>

using namespace std;

// Лёгкое представление матрицы (не владеет памятью)
// Хранение по строкам (row-major): элемент [i][j] = data_[i * ncols_ + j]
template <typename AtomT>
class MatrixView {
private:
    AtomT* data_;
    int nrows_;
    int ncols_;

public:
    __host__ __device__ MatrixView(AtomT* data, int nrows, int ncols)
        : data_(data), nrows_(nrows), ncols_(ncols) {}

    __host__ __device__ int nrows() const { return nrows_; }
    __host__ __device__ int ncols() const { return ncols_; }
    __host__ __device__ int size()  const { return nrows_ * ncols_; }

    // Линейный доступ к элементу по плоскому индексу
    __host__ __device__ AtomT& operator[](int n) { return data_[n]; }
    __host__ __device__ const AtomT& operator[](int n) const { return data_[n]; }

    // Двумерный доступ: строка i, столбец j
    __host__ __device__ AtomT& operator()(int i, int j) { return data_[i * ncols_ + j]; }
    __host__ __device__ const AtomT& operator()(int i, int j) const { return data_[i * ncols_ + j]; }
};
