#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <initializer_list>
#include <algorithm> // Needed for std::copy
#include <vector>

namespace BaseML
{
    class Matrix
    {
    private:
        size_t rows, cols;
        float* data;

    public:
        // Default constructor for creating an empty object
        Matrix(); 

        // Create an empty Matrix with 'numOfRows' rows and 'numOfCols' columns
        Matrix(size_t numOfRows, size_t numOfCols);

        // Create a Matrix from a 2 dimensional initializer list. The length of every 
        // row in the initializer list should be the same. Set 'transposed' to true to 
        // make every inner initializer list a column instead of a row in the Matrix
        Matrix(std::initializer_list<std::initializer_list<float>> init, bool transposed = false);

        // Create a Matrix from a single dimesional initializer list.
        // The Matrix will have only one row / column. The Matrix will have one column 
        // by default. To change this and make it a row vector instead, set 'columnVector' 
        // to 'false'.
        Matrix(std::initializer_list<float> init, bool columnVector = true);

        // Create a Matrix from a 2 dimensional vector. The length of every 
        // row in the vector should be the same. Set 'transposed' to true to 
        // make every inner vector a column instead of a row in the Matrix
        Matrix(std::vector<std::vector<float>>& vec, bool transposed = false);

        // Create a Matrix from a single dimesional vector.
        // The Matrix will have only one row / column. The Matrix will have one column 
        // by default. To change this and make it a row vector instead, set 'columnVector' 
        // to 'false'.
        Matrix(std::vector<float>& vec, bool columnVector = true);

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
        float& operator()(size_t row, size_t col); 

        // Const access operator
        const float& operator()(size_t row, size_t col) const; 

        // 1D access operator
        float& operator()(size_t index);

        // 1D const access operator
        const float& operator()(size_t index) const;

        // Matrix addition. This function assumes that the matrices have the 
        // same size.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix operator+(const Matrix& other) const;

        // Matrix subtraction. This function assumes that the matrices have the 
        // same size.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix operator-(const Matrix& other) const;

        // Multiply the matrix by a scalar.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix operator*(float operand) const;

        // Matrix multiplication. This function assumes that the sizes of the matrices 
        // are compatible with each other.
        // Two matrices are compatible only if mat1.columnsCount() == mat2.rowsCount().
        // Warning: this function doesn't check for the correctness of the input!
        Matrix operator*(const Matrix& other) const;

        // Functions

        // Returns the number of rows in the Matrix
        size_t rowsCount() const;

        // Returns the number of columns in the Matrix
        size_t columnsCount() const;

        // Returns the number of elements in the Matrix
        size_t size() const;

        // Returns the transposition of the Matrix
        Matrix transpose() const;

        // Add a column vector to each column of the Matrix.
        // The Matrix and the column vector should have the same number of rows and 
        // the column vector should have only one coulumn (should be a Matrix with 
        // one column).
        // Warning: this function doesn't check for the correctness of the input!
        Matrix& addToColumns(const Matrix& columnVec);

        // Sum each row of the Matrix and return a column vector (Matrix 
        // with one column) with each of the rows' sum.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix sumRows() const;

        // Matrix elementwise multiplication. This function multiplies each element in 
        // this Matrix with the corresponding element of the other Matrix and returns 
        // the result. This function assumes that the matrices have the same size.
        // Warning: this function doesn't check for the correctness of the input!
        Matrix multElementwise(const Matrix& other) const;

        // Apply the given function on every element of the Matrix
        void applyToElements(float (*func)(float));

        // Replaces all of the values of the Matrix with 0
        void clear();

        // Save Matrix to disk. Assumes a binary output stream
        void save(std::ofstream& outFile);

        // Load Matrix from disk. Assumes a binary input stream
        void load(std::ifstream& inFile);

        // Prints the Matrix to the console
        void print() const;
    };
}