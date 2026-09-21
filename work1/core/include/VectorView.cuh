#pragma once

#include <cuda_runtime.h>

using namespace std;

// Легковесное представление вектора (не владеет памятью)
template <typename AtomT>
class VectorView {
private:
    AtomT* data_;
    int size_;

public:
    __host__ __device__ VectorView(AtomT* data, int size) : data_(data), size_(size) {}

    __host__ __device__ int size() const { return size_; }

    __host__ __device__ AtomT& operator[](int n) { return data_[n]; }
    __host__ __device__ const AtomT& operator[](int n) const { return data_[n]; }

    __host__ __device__ AtomT& operator()(int i) { return data_[i]; }
    __host__ __device__ const AtomT& operator()(int i) const { return data_[i]; }
};
