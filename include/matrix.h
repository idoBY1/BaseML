#pragma once

#include <algorithm> // Needed for std::copy

namespace MachineLearning
{
    template<typename T>
    class Matrix
    {
    private:
        size_t rows, cols;
        T* data;

    public:
        Matrix(); // Default constructor for creating an empty object
        Matrix(size_t numOfRows, size_t numOfCols);
        ~Matrix();

        Matrix(const Matrix& other); // Copy contructor
        Matrix(Matrix&& other) noexcept; // Move constructor

        // operators overload (operators are invoked when changing an object that already exists)
        Matrix& operator=(const Matrix& other); // Copy operator
        Matrix& operator=(Matrix&& other) noexcept; // Move operator

        T& operator()(size_t row, size_t col); // access operator
        const T& operator()(size_t row, size_t col) const; // const access operator

        size_t getRows();
        size_t getColumns();
    };

    // --- IMPLEMENTATION ---

    template<typename T>
    inline Matrix<T>::Matrix()
        :rows(0), cols(0), data(nullptr)
    {
    }

    template<typename T>
    Matrix<T>::Matrix(size_t numOfRows, size_t numOfCols)
        : rows(numOfRows), cols(numOfCols)
    {
        data = new T[rows * cols];
    }

    template<typename T>
    Matrix<T>::~Matrix()
    {
        delete[] data;
    }

    template<typename T>
    Matrix<T>::Matrix(const Matrix<T>& other)
        : rows(other.rows), cols(other.cols)
    {
        data = new T[rows * cols];
        std::copy(other.data, other.data + (rows * cols), data);
    }

    template<typename T>
    Matrix<T>::Matrix(Matrix<T>&& other) noexcept
        : rows(other.rows), cols(other.cols), data(other.data)
    {
        other.data = nullptr; // Nullify the pointer to avoid double deletion
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
    {
        if (this != &other) {
            delete[] data; // Free existing memory

            rows = other.rows;
            cols = other.cols;
            data = new T[rows * cols]; // Allocate new memory

            std::copy(other.data, other.data + (rows * cols), data);
        }

        return *this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept
    {
        if (this != &other) {
            delete[] data; // Free existing memory

            rows = other.rows;
            cols = other.cols;
            data = other.data;

            other.data = nullptr; // Nullify the pointer to avoid double deletion
        }

        return *this;
    }

    template<typename T>
    T& Matrix<T>::operator()(size_t row, size_t col)
    {
        return data[row * cols + col];
    }

    template<typename T>
    const T& Matrix<T>::operator()(size_t row, size_t col) const
    {
        return data[row * cols + col];
    }

    template<typename T>
    inline size_t Matrix<T>::getRows()
    {
        return rows;
    }

    template<typename T>
    inline size_t Matrix<T>::getColumns()
    {
        return cols;
    }
}