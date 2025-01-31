#include "matrix.h"

namespace BaseML
{
    Matrix::Matrix()
        :rows(0), cols(0), data(nullptr)
    {
    }

    Matrix::Matrix(size_t numOfRows, size_t numOfCols)
        : rows(numOfRows), cols(numOfCols)
    {
        data = new float[rows * cols];
    }

    Matrix::Matrix(std::initializer_list<std::initializer_list<float>> init, bool transposed)
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

            data = new float[rows * cols];

            size_t row = 0;
            for (const auto& innerList : init) {
                size_t col = 0;
                for (const float& value : innerList) {
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

            data = new float[rows * cols];

            size_t col = 0;
            for (const auto& innerList : init) {
                size_t row = 0;
                for (const float& value : innerList) {
                    data[row * cols + col] = value;
                    row++;
                }
                col++;
            }
        }
    }

    Matrix::Matrix(std::initializer_list<float> init, bool columnVector)
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

        data = new float[rows * cols];

        std::copy(init.begin(), init.end(), data);
    }

    Matrix::Matrix(std::vector<std::vector<float>>& vec, bool transposed)
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

            data = new float[rows * cols];

            size_t row = 0;
            for (const auto& innerVec : vec) {
                size_t col = 0;
                for (const float& value : innerVec) {
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

            data = new float[rows * cols];

            size_t col = 0;
            for (const auto& innerVec : vec) {
                size_t row = 0;
                for (const float& value : innerVec) {
                    data[row * cols + col] = value;
                    row++;
                }
                col++;
            }
        }
    }

    Matrix::Matrix(std::vector<float>& vec, bool columnVector)
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

        data = new float[rows * cols];

        std::copy(vec.begin(), vec.end(), data);
    }

    Matrix::~Matrix()
    {
        delete[] data;
    }

    Matrix::Matrix(const Matrix& other)
        : rows(other.rows), cols(other.cols)
    {
        data = new float[rows * cols];
        std::copy(other.data, other.data + (rows * cols), data);
    }

    Matrix::Matrix(Matrix&& other) noexcept
        : rows(other.rows), cols(other.cols), data(other.data)
    {
        other.data = nullptr; // Nullify the pointer to avoid double deletion
    }

    Matrix& Matrix::operator=(const Matrix& other)
    {
        if (this != &other) {
            delete[] data; // Free existing memory

            rows = other.rows;
            cols = other.cols;
            data = new float[rows * cols]; // Allocate new memory

            std::copy(other.data, other.data + (rows * cols), data);
        }

        return *this;
    }

    Matrix& Matrix::operator=(Matrix&& other) noexcept
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

    float& Matrix::operator()(size_t row, size_t col)
    {
        return data[row * cols + col];
    }

    const float& Matrix::operator()(size_t row, size_t col) const
    {
        return data[row * cols + col];
    }

    float& Matrix::operator()(size_t index)
    {
        return data[index];
    }

    const float& Matrix::operator()(size_t index) const
    {
        return data[index];
    }

    Matrix Matrix::operator+(const Matrix& other) const
    {
        Matrix newMat(rows, cols);

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

    Matrix Matrix::operator-(const Matrix& other) const
    {
        Matrix newMat(rows, cols);

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

    Matrix Matrix::operator*(float operand) const
    {
        Matrix newMat(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                newMat(i, j) = (*this)(i, j) * operand;
            }
        }

        return newMat;
    }

    Matrix Matrix::operator*(const Matrix& other) const
    {
        Matrix newMat(rows, other.cols);

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

    size_t Matrix::rowsCount() const
    {
        return rows;
    }

    size_t Matrix::columnsCount() const
    {
        return cols;
    }

    size_t Matrix::size() const
    {
        return rows * cols;
    }

    Matrix Matrix::transpose() const
    {
        Matrix newMat(cols, rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                newMat(j, i) = (*this)(i, j);
            }
        }

        return newMat;
    }

    Matrix& Matrix::addToColumns(const Matrix& columnVec)
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

    Matrix Matrix::sumRows() const
    {
        Matrix newMat(rows, 1);

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

    void Matrix::save(std::ofstream& outFile)
    {
        outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        outFile.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(float));
    }

    void Matrix::load(std::ifstream& inFile)
    {
        delete[] data; // Free existing memory

        inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        data = new float[rows * cols]; // Allocate memory for data from disk

        inFile.read(reinterpret_cast<char*>(data), rows * cols * sizeof(float));
    }

    void Matrix::print() const
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