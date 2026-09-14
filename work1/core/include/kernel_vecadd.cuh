#pragma once

#include <cstddef>
#include "VectorView.cuh"

using namespace std;

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> lhs, VectorView<AtomT> rhs, VectorView<AtomT> out) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < out.size()) {
        out[i] = lhs[i] + rhs[i];
    }
}

