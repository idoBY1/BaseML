#pragma once

#include <iostream>
#include <initializer_list>
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
        // Default constructor for creating an empty object
        Matrix(); 

        // Create an empty Matrix with 'numOfRows' rows and 'numOfCols' columns
        Matrix(size_t numOfRows, size_t numOfCols);

        // Create a Matrix from a 2 dimensional initializer list.
        // The length of every row in the initializer list should be the same.
        Matrix(std::initializer_list<std::initializer_list<T>> init);

        // Create a Matrix from a single dimesional initializer list.
        // The Matrix will have only one row / column. The Matrix will have one column 
        // by default. To change this and make it a row vector instead, set 'columnVector' 
        // to 'false'.
        Matrix(std::initializer_list<T> init, bool columnVector = true);

        // Destructor
        ~Matrix();

        // Copy contructor
        Matrix(const Matrix& other); 

        // Move constructor
        Matrix(Matrix&& other) noexcept; 

        // Operators overload (operators are invoked when changing an object that already exists)

        // Copy operator
        Matrix& operator=(const Matrix& other); 

        // Move operator
        Matrix& operator=(Matrix&& other) noexcept; 

        // Access operator
        T& operator()(size_t row, size_t col); 

        // Const access operator
        const T& operator()(size_t row, size_t col) const; 

        // 1D access operator
        T& operator()(size_t index);

        // 1D const access operator
        const T& operator()(size_t index) const;

        // Matrix addition. This function assumes that the matrices have the 
        // same size and that they both contain float types.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T> operator+(const Matrix<T>& other);

        // Multiply the matrix by a scalar. This function assumes that the Matrix 
        // contains float types.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T> operator*(T operand);

        // Matrix multiplication. This function assumes that the matrices contain 
        // floats and that the sizes of the matrices are compatible with each other.
        // Two matrices are compatible only if mat1.columnsCount() == mat2.rowsCount().
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T> operator*(const Matrix<T>& other);

        // Functions

        // Returns the number of rows in the Matrix
        size_t rowsCount() const;

        // Returns the number of columns in the Matrix
        size_t columnsCount() const;

        // Returns the number of elements in the Matrix
        size_t size() const;

        // Prints the Matrix to the console
        void print() const;
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
    Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init)
    {
        rows = init.size();
        cols = init.begin()->size();

        // Verify that inner lists are of equal size
        for (const auto& innerList : init) {
            if (innerList.size() != cols) {
                throw std::runtime_error("Inner initializer lists must have equal size");
            }
        }

        data = new T[rows * cols];

        size_t row = 0;
        for (const auto& innerList : init) {
            size_t col = 0;
            for (const T& value : innerList) {
                data[row * cols + col] = value;
                col++;
            }
            row++;
        }
    }

    template<typename T>
    Matrix<T>::Matrix(std::initializer_list<T> init, bool columnVector)
    {
        if (columnVector)
        {
            rows = init.size();
            cols = 1;
        }
        else
        {
            rows = 1;
            cols = init.size();
        }

        data = new T[rows * cols];

        std::copy(init.begin(), init.end(), data);
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
    inline T& Matrix<T>::operator()(size_t row, size_t col)
    {
        return data[row * cols + col];
    }

    template<typename T>
    inline const T& Matrix<T>::operator()(size_t row, size_t col) const
    {
        return data[row * cols + col];
    }

    template<typename T>
    inline T& Matrix<T>::operator()(size_t index)
    {
        return data[index];
    }

    template<typename T>
    inline const T& Matrix<T>::operator()(size_t index) const
    {
        return data[index];
    }

    template<>
    inline Matrix<float> Matrix<float>::operator+(const Matrix<float>& other)
    {
        Matrix<float> newMat(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                newMat(i, j) = (*this)(i, j) + other(i, j);
            }
        }

        return newMat;
    }

    template<>
    inline Matrix<float> Matrix<float>::operator*(float operand)
    {
        Matrix<float> newMat(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                newMat(i, j) = (*this)(i, j) * operand;
            }
        }

        return newMat;
    }

    template<>
    inline Matrix<float> Matrix<float>::operator*(const Matrix<float>& other)
    {
        Matrix<float> newMat(rows, other.cols);

        for (int i = 0; i < newMat.rows; i++)
        {
            for (int j = 0; j < newMat.cols; j++)
            {
                newMat(i, j) = 0.0f;

                for (int k = 0; k < cols; k++)
                {
                    newMat(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }

        return newMat;
    }

    template<typename T>
    inline size_t Matrix<T>::rowsCount() const
    {
        return rows;
    }

    template<typename T>
    inline size_t Matrix<T>::columnsCount() const
    {
        return cols;
    }

    template<typename T>
    inline size_t Matrix<T>::size() const
    {
        return rows * cols;
    }

    template<typename T>
    void Matrix<T>::print() const
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout << (*this)(i, j) << "  ";
            }
            std::cout << "\n";
        }

        std::cout << std::endl;
    }
}