#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <cstddef>

template <typename AtomT>
class Data {
private:
    std::size_t size_;
    AtomT* data_;

    void allocate(std::size_t size) {
        if (size > 0) {
            cudaError_t err = cudaMalloc(&data_, size * sizeof(AtomT));
            if (err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }
        } else {
            data_ = nullptr;
        }
    }

    void free_memory() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

public:
    explicit Data(std::size_t size) : size_(size), data_(nullptr) {
        allocate(size_);
    }

    Data(const Data& other) : size_(other.size_), data_(nullptr) {
        allocate(size_);
        if (size_ > 0) {
            cudaError_t err = cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                free_memory();
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    Data(Data&& other) noexcept : size_(other.size_), data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr;
    }

    Data& operator=(const Data& other) {
        if (this != &other) {
            Data temp(other);
            std::swap(size_, temp.size_);
            std::swap(data_, temp.data_);
        }
        return *this;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            free_memory();
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    AtomT* data() { return data_; }
    const AtomT* data() const { return data_; }
    std::size_t size() const { return size_; }

    void copy_to_host(AtomT* host_ptr) const {
        if (size_ > 0) {
            cudaError_t err = cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    void copy_from_host(const AtomT* host_ptr) {
        if (size_ > 0) {
            cudaError_t err = cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    ~Data() {
        free_memory();
    }
};

