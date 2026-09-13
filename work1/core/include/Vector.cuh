#pragma once

#include "Data.cuh"
#include "VectorView.cuh"
#include <memory>
#include <cstddef>

template <typename AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    explicit Vector(std::size_t size) 
        : data_(std::make_shared<Data<AtomT>>(size)),
          view_(data_->data(), size) {}

    // Optionally allow creating from existing Data to support views of subsets,
    // but assignment just says Vector(size).
    // Let's stick to the diagram.

    std::size_t size() const {
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

// operator+ declaration
template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs);

// Include kernel to define operator+
#include "kernel_vecadd.cuh"
#include <stdexcept>

template <typename AtomT>
void Vector<AtomT>::add_from(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size() || size() != lhs.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    int blockSize = 256;
    int numBlocks = (lhs.size() + blockSize - 1) / blockSize;

    if (lhs.size() > 0) {
        kernel_vecadd<<<numBlocks, blockSize>>>(lhs.view(), rhs.view(), view_);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs) {
    Vector<AtomT> result(lhs.size());
    result.add_from(lhs, rhs);

    return result;
}
