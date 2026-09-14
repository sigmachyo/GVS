#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "Data.cuh"
#include "VectorView.cuh"
#include "kernel_vecadd.cuh"

using namespace std;

template <typename AtomT>
class Vector {
private:
    shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    explicit Vector(size_t size) 
        : data_(make_shared<Data<AtomT>>(size)),
          view_(data_->data(), size) {}

    size_t size() const {
        return view_.size();
    }

    Data<AtomT>& data() {
        return *data_;
    }

    const Data<AtomT>& data() const {
        return *data_;
    }

    VectorView<AtomT>& view() {
        return view_;
    }

    const VectorView<AtomT>& view() const {
        return view_;
    }

    void add_from(const Vector& lhs, const Vector& rhs);
};

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs);

template <typename AtomT>
void Vector<AtomT>::add_from(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size() || size() != lhs.size()) {
        throw invalid_argument("Vectors must have the same size");
    }

    size_t blockSize = 256;
    size_t numBlocks = (lhs.size() + blockSize - 1) / blockSize;

    if (lhs.size() > 0) {
        kernel_vecadd<<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(blockSize)>>>(lhs.view(), rhs.view(), view_);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(err));
        }
    }
}

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs) {
    Vector<AtomT> result(lhs.size());
    result.add_from(lhs, rhs);

    return result;
}
