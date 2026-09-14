#pragma once

#include <cstddef>
#include <cuda_runtime.h>

using namespace std;

template <typename AtomT>
class VectorView {
private:
    AtomT* data_;
    size_t size_;

public:
    __host__ __device__ VectorView(AtomT* data, size_t size) : data_(data), size_(size) {}
    __host__ __device__ VectorView() : data_(nullptr), size_(0) {}

    __host__ __device__ size_t size() const {
        return size_;
    }

    __host__ __device__ AtomT& operator[](size_t n) {
        return data_[n];
    }

    __host__ __device__ const AtomT& operator[](size_t n) const {
        return data_[n];
    }

    __host__ __device__ AtomT& operator()(size_t i) {
        return data_[i];
    }

    __host__ __device__ const AtomT& operator()(size_t i) const {
        return data_[i];
    }
};

