#pragma once

#include "MatrixView.cuh"

// CUDA-ядро блочного умножения матриц с разделяемой памятью: C = A * B
// Входные матрицы a и b передаются как MatrixView<const AtomT>, так как они доступны только для чтения
template <typename AtomT, int BLOCK_SIZE = 16>
__global__ void kernel_matmul_shmem(const MatrixView<const AtomT> a, const MatrixView<const AtomT> b, MatrixView<AtomT> c) {
    // Буферы в разделяемой памяти для тайлов матриц A и B
    __shared__ AtomT As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ AtomT Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    AtomT sum = 0;

    // Количество тайлов вдоль общего измерения k
    int num_tiles = (a.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // 1. Кооперативная загрузка тайла матрицы A в разделяемую память
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        if (row < a.nrows() && a_col < a.ncols()) {
            As[threadIdx.y][threadIdx.x] = a(row, a_col);
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // 2. Кооперативная загрузка тайла матрицы B в разделяемую память
        int b_row = t * BLOCK_SIZE + threadIdx.y;
        if (b_row < b.nrows() && col < b.ncols()) {
            Bs[threadIdx.y][threadIdx.x] = b(b_row, col);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        // Барьер: ждём завершения загрузки тайлов всеми нитями блока
        __syncthreads();

        // 3. Вычисление частичного произведения элементов тайлов
        #pragma unroll
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            sum += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        }

        // Барьер: ждём, пока все нити закончат вычисления с текущими тайлами
        __syncthreads();
    }

    // 4. Запись результата в глобальную память (только если координаты внутри матрицы C)
    if (row < c.nrows() && col < c.ncols()) {
        c(row, col) = sum;
    }
}
