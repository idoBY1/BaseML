#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <initializer_list>
#include <algorithm> // Needed for std::copy
#include <vector>

namespace BaseML
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

        // Create a Matrix from a 2 dimensional initializer list. The length of every 
        // row in the initializer list should be the same. Set 'transposed' to true to 
        // make every inner initializer list a column instead of a row in the Matrix
        Matrix(std::initializer_list<std::initializer_list<T>> init, bool transposed = false);

        // Create a Matrix from a single dimesional initializer list.
        // The Matrix will have only one row / column. The Matrix will have one column 
        // by default. To change this and make it a row vector instead, set 'columnVector' 
        // to 'false'.
        Matrix(std::initializer_list<T> init, bool columnVector = true);

        // Create a Matrix from a 2 dimensional vector. The length of every 
        // row in the vector should be the same. Set 'transposed' to true to 
        // make every inner vector a column instead of a row in the Matrix
        Matrix(std::vector<std::vector<T>>& vec, bool transposed = false);

        // Create a Matrix from a single dimesional vector.
        // The Matrix will have only one row / column. The Matrix will have one column 
        // by default. To change this and make it a row vector instead, set 'columnVector' 
        // to 'false'.
        Matrix(std::vector<T>& vec, bool columnVector = true);

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
        Matrix<T> operator+(const Matrix<T>& other) const;

        // Matrix subtraction. This function assumes that the matrices have the 
        // same size and that they both contain float types.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T> operator-(const Matrix<T>& other) const;

        // Multiply the matrix by a scalar. This function assumes that the Matrix 
        // contains float types.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T> operator*(T operand) const;

        // Matrix multiplication. This function assumes that the matrices contain 
        // floats and that the sizes of the matrices are compatible with each other.
        // Two matrices are compatible only if mat1.columnsCount() == mat2.rowsCount().
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T> operator*(const Matrix<T>& other) const;

        // Functions

        // Returns the number of rows in the Matrix
        size_t rowsCount() const;

        // Returns the number of columns in the Matrix
        size_t columnsCount() const;

        // Returns the number of elements in the Matrix
        size_t size() const;

        // Returns the transposition of the Matrix
        Matrix<T> transpose() const;

        // Add a column vector to each column of the Matrix.
        // The Matrix and the column vector should have the same number of rows and 
        // the column vector should have only one coulumn (should be a Matrix with 
        // one column).
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T>& addToColumns(const Matrix<T>& columnVec);

        // Sum each row of the Matrix and return a column vector (Matrix 
        // with one column) with each of the rows' sum.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix<T> sumRows() const;

        // Save Matrix to disk. Assumes a binary output stream
        void save(std::ofstream& outFile);

        // Load Matrix from disk. Assumes a binary input stream
        void load(std::ifstream& inFile);

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
    Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init, bool transposed)
    {
        if (!transposed)
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
        else
        {
            rows = init.begin()->size();
            cols = init.size();

            // Verify that inner lists are of equal size
            for (const auto& innerList : init) {
                if (innerList.size() != rows) {
                    throw std::runtime_error("Inner initializer lists must have equal size");
                }
            }

            data = new T[rows * cols];

            size_t col = 0;
            for (const auto& innerList : init) {
                size_t row = 0;
                for (const T& value : innerList) {
                    data[row * cols + col] = value;
                    row++;
                }
                col++;
            }
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
    inline Matrix<T>::Matrix(std::vector<std::vector<T>>& vec, bool transposed)
    {
        if (!transposed)
        {
            rows = vec.size();
            cols = vec.begin()->size();

            // Verify that inner lists are of equal size
            for (const auto& innerVec : vec) {
                if (innerVec.size() != cols) {
                    throw std::runtime_error("Inner vectors must have equal size");
                }
            }

            data = new T[rows * cols];

            size_t row = 0;
            for (const auto& innerVec : vec) {
                size_t col = 0;
                for (const T& value : innerVec) {
                    data[row * cols + col] = value;
                    col++;
                }
                row++;
            }
        }
        else
        {
            rows = vec.begin()->size();
            cols = vec.size();

            // Verify that inner lists are of equal size
            for (const auto& innerVec : vec) {
                if (innerVec.size() != rows) {
                    throw std::runtime_error("Inner vectors must have equal size");
                }
            }

            data = new T[rows * cols];

            size_t col = 0;
            for (const auto& innerVec : vec) {
                size_t row = 0;
                for (const T& value : innerVec) {
                    data[row * cols + col] = value;
                    row++;
                }
                col++;
            }
        }
    }

    template<typename T>
    inline Matrix<T>::Matrix(std::vector<T>& vec, bool columnVector)
    {
        if (columnVector)
        {
            rows = vec.size();
            cols = 1;
        }
        else
        {
            rows = 1;
            cols = vec.size();
        }

        data = new T[rows * cols];

        std::copy(vec.begin(), vec.end(), data);
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
    inline Matrix<float> Matrix<float>::operator+(const Matrix<float>& other) const
    {
        Matrix<float> newMat(rows, cols);

#ifdef DEBUG
        if (size() != other.size())
        {
            std::cout << "Invalid sizes in Matrix addition" << std::endl;
            throw std::runtime_error("Invalid matrix addition");
        }
#endif // DEBUG

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
    inline Matrix<float> Matrix<float>::operator-(const Matrix<float>& other) const
    {
        Matrix<float> newMat(rows, cols);

#ifdef DEBUG
        if (size() != other.size())
        {
            std::cout << "Invalid sizes in Matrix subtraction" << std::endl;
            throw std::runtime_error("Invalid matrix subtraction");
        }
#endif // DEBUG

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                newMat(i, j) = (*this)(i, j) - other(i, j);
            }
        }

        return newMat;
    }

    template<>
    inline Matrix<float> Matrix<float>::operator*(float operand) const
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
    inline Matrix<float> Matrix<float>::operator*(const Matrix<float>& other) const
    {
        Matrix<float> newMat(rows, other.cols);

#ifdef DEBUG
        if (cols != other.rows)
        {
            std::cout << "Invalid sizes in Matrix multiplication: c=" << cols 
                << " r=" << other.rows << std::endl;
            throw std::runtime_error("Invalid matrix multiplication");
        }
#endif // DEBUG

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
    inline Matrix<T> Matrix<T>::transpose() const
    {
        Matrix<T> newMat(cols, rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                newMat(j, i) = (*this)(i, j);
            }
        }

        return newMat;
    }

    template<>
    inline Matrix<float>& Matrix<float>::addToColumns(const Matrix<float>& columnVec)
    {
#ifdef DEBUG
        if (columnVec.cols != 1 || rows != columnVec.rows)
        {
            std::cout << "Invalid call to addToColumns(columnVec)" << std::endl;
            throw std::runtime_error("Invalid call");
        }
#endif // DEBUG

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                (*this)(i, j) += columnVec(i);
            }
        }
        
        return (*this);
    }

    template<>
    inline Matrix<float> Matrix<float>::sumRows() const
    {
        Matrix<float> newMat(rows, 1);

        for (int i = 0; i < rows; i++)
        {
            newMat(i, 0) = 0.0f;

            for (int j = 0; j < cols; j++)
            {
                newMat(i, 0) += (*this)(i, j);
            }
        }

        return newMat;
    }

    template<typename T>
    inline void Matrix<T>::save(std::ofstream& outFile)
    {
        outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        outFile.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(T));
    }

    template<typename T>
    inline void Matrix<T>::load(std::ifstream& inFile)
    {
        delete[] data; // Free existing memory

        inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        data = new T[rows * cols]; // Allocate memory for data from disk

        inFile.read(reinterpret_cast<char*>(data), rows * cols * sizeof(T));
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