#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

using namespace std;

// Класс-обёртка RAII для памяти GPU
template <typename AtomT>
class Data {
private:
    int size_;
    AtomT* data_;

public:
    // Конструктор: выделяем память на GPU
    explicit Data(int size) : size_(size), data_(nullptr) {
        if (size_ > 0) {
            cudaError_t err = cudaMalloc(&data_, size_ * sizeof(AtomT));
            if (err != cudaSuccess)
                throw runtime_error(cudaGetErrorString(err));
        }
    }

    // Конструктор копирования: выделяем новую память и копируем данные (D2D)
    Data(const Data& other) : size_(other.size_), data_(nullptr) {
        if (size_ > 0) {
            cudaMalloc(&data_, size_ * sizeof(AtomT));
            cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
        }
    }

    // Конструктор перемещения: забираем указатель у other
    Data(Data&& other) noexcept : size_(other.size_), data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr;
    }

    // Оператор присваивания копированием
    Data& operator=(const Data& other) {
        if (this != &other) {
            if (data_) cudaFree(data_);
            size_ = other.size_;
            data_ = nullptr;
            if (size_ > 0) {
                cudaMalloc(&data_, size_ * sizeof(AtomT));
                cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    // Оператор присваивания перемещением
    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    // Деструктор: освобождаем память GPU
    ~Data() {
        if (data_) cudaFree(data_);
    }

    AtomT* data() { return data_; }
    const AtomT* data() const { return data_; }
    int size() const { return size_; }

    // Копирование данных с GPU на CPU
    void copy_to_host(AtomT* host_ptr) const {
        if (size_ > 0)
            cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost);
    }

    // Копирование данных с CPU на GPU
    void copy_from_host(const AtomT* host_ptr) {
        if (size_ > 0)
            cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice);
    }
};
