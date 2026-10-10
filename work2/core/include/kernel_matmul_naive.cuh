#pragma once

#include "MatrixView.cuh"

// CUDA-ядро наивного умножения матриц: C = A * B
// Каждый поток вычисляет один элемент результирующей матрицы C
template <typename AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ: лишние потоки не должны выходить за пределы матрицы
    if (row < c.nrows() && col < c.ncols()) {
        AtomT sum = 0;
        for (int k = 0; k < a.ncols(); ++k) {
            sum += a(row, k) * b(k, col);
        }
        c(row, col) = sum;
    }
}
