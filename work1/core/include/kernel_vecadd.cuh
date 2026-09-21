#pragma once

#include "VectorView.cuh"

// CUDA-ядро для поэлементного сложения двух векторов
template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> lhs, VectorView<AtomT> rhs, VectorView<AtomT> out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверка границы, чтобы лишние потоки не вышли за пределы массива
    if (i < out.size()) {
        out[i] = lhs[i] + rhs[i];
    }
}
