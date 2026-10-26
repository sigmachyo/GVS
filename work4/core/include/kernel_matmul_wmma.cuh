#pragma once

#include <cuda_fp16.h>
#include <mma.h>
#include "MatrixView.cuh"

using namespace nvcuda;

// WMMA-тайл: 16×16×16 (M×N×K) — фиксированный размер фрагмента тензорного ядра
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// CUDA-ядро умножения матриц с использованием тензорных ядер (WMMA API): C = A * B
// Блок (128, 4) = 512 нитей = 16 варпов, расположенных в сетке 4×4 WMMA-тайлов (64×64 элементов)
// Входные матрицы хранят __half (FP16), результат накапливается в float (FP32),
// затем конвертируется обратно в __half при записи
__global__ void kernel_matmul_wmma(const MatrixView<const __half> a,
                                   const MatrixView<const __half> b,
                                   MatrixView<__half> c) {
    // Определяем позицию варпа внутри блока:
    // threadIdx.x / 32 — столбец варпа (0..3), threadIdx.y — строка варпа (0..3)
    int warpCol = threadIdx.x / 32;   // 0..3
    int warpRow = threadIdx.y;         // 0..3

    // Глобальные координаты WMMA-тайла результирующей матрицы C
    int warpsPerBlockX = blockDim.x / 32;  // 4
    int warpM = (blockIdx.y * blockDim.y + warpRow) * WMMA_M;
    int warpN = (blockIdx.x * warpsPerBlockX + warpCol) * WMMA_N;

    // Проверка границ: варп может выйти за пределы матрицы C
    if (warpM >= c.nrows() || warpN >= c.ncols()) return;

    // Объявляем фрагмент-аккумулятор (float) и обнуляем его
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Количество тайлов вдоль общего измерения k
    int num_tiles = (a.ncols() + WMMA_K - 1) / WMMA_K;

    for (int t = 0; t < num_tiles; ++t) {
        int aCol = t * WMMA_K;
        int bRow = t * WMMA_K;

        // Загрузка фрагмента матрицы A (WMMA_M × WMMA_K) из глобальной памяти
        // Stride (leading dimension) = a.ncols() для row-major
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        if (warpM + WMMA_M <= a.nrows() && aCol + WMMA_K <= a.ncols()) {
            wmma::load_matrix_sync(a_frag, a.data() + warpM * a.ncols() + aCol, a.ncols());
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }

        // Загрузка фрагмента матрицы B (WMMA_K × WMMA_N) из глобальной памяти
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
        if (bRow + WMMA_K <= b.nrows() && warpN + WMMA_N <= b.ncols()) {
            wmma::load_matrix_sync(b_frag, b.data() + bRow * b.ncols() + warpN, b.ncols());
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }

        // Матричное умножение-сложение на тензорном ядре: acc += a_frag * b_frag
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Сохранение результата: конвертируем float-аккумулятор в __half и записываем в C
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_half;
    for (int i = 0; i < acc.num_elements; ++i) {
        acc_half.x[i] = __float2half(acc.x[i]);
    }

    if (warpM + WMMA_M <= c.nrows() && warpN + WMMA_N <= c.ncols()) {
        wmma::store_matrix_sync(c.data() + warpM * c.ncols() + warpN, acc_half, c.ncols(), wmma::mem_row_major);
    }
}
