#pragma once

#include "VectorView.cuh"

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> lhs, VectorView<AtomT> rhs, VectorView<AtomT> out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out.size()) {
        out[i] = lhs[i] + rhs[i];
    }
}

