#include "utils.h"

#include <random>
#include <iostream>
#include <iomanip>

#include "matrix.h"

namespace BaseML::Utils
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

    float leakyReLU(float input)
    {
        if (input > 0.0f)
            return input;
        else
            return 0.01f * input;
    }

    float leakyReLUDerivative(float neuronOutput)
    {
        if (neuronOutput > 0.0f)
            return 1.0f;
        else
            return 0.01f;
    }

    float BaseML::Utils::squareError(float activation, float expected)
    {
        float error = activation - expected;

        return error * error;
    }

    float squareErrorDerivative(float activation, float expected)
    {
        return 2 * (activation - expected);
    }
}