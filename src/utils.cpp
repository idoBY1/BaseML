#include "utils.h"

#include <random>
#include <iostream>
#include <iomanip>

#include "matrix.h"

namespace MachineLearning::Utils
{
    float getRandomFloat(float min, float max)
    {
        // Initialize the random number generator engine with a seed
        // Declared as static to initialize the seed only once
        static std::random_device rd;
        static std::mt19937 gen(rd()); // Use Mersenne Twister engine

        std::uniform_real_distribution<float> distrib(min, max);
        return distrib(gen);
    }

    float sigmoid(float input)
    {
        return 1.0f / (1.0f + expf(-input));
    }

    // (the derivative of the sigmoid function can be calculated from its output)
    float sigmoidDerivative(float neuronOutput)
    {
        return neuronOutput * (1.0f - neuronOutput);
    }

    float MachineLearning::Utils::squareError(float activation, float expected)
    {
        float error = activation - expected;

        return error * error;
    }

    void printFloatMatrix(const Matrix<float>& mat)
    {
        for (int i = 0; i < mat.rowsCount(); i++)
        {
            for (int j = 0; j < mat.columnsCount(); j++)
            {
                std::cout << std::setw(8) << std::setprecision(5) << mat(i, j) << " ";
            }
            std::cout << "\n";
        }

        std::cout << std::endl;
    }
}