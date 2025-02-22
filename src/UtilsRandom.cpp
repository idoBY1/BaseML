#include "UtilsRandom.h"

#include <random>
#include <cmath>

namespace BaseML::Utils
{
    GaussianSampler::GaussianSampler(float stddev)
        :gen(std::random_device{}()), distrib(0.0f, stddev)
    {
    }

    float GaussianSampler::sample(float mean)
    {
        return mean + distrib(gen);
    }

    float GaussianSampler::logProbabiltiy(float mean, float sample)
    {
        float diff = sample - mean;
        return -std::log(distrib.stddev() * std::sqrt(2.0f * PI)) - 0.5f * (diff * diff) / (distrib.stddev() * distrib.stddev());
    }

    float getRandomFloat(float min, float max)
    {
        // Initialize the random number generator engine with a seed
        // Declared as static to initialize the seed only once
        static std::random_device rd;
        static std::mt19937 gen(rd()); // Use Mersenne Twister engine

        std::uniform_real_distribution<float> distrib(min, max);
        return distrib(gen);
    }

    float initFromNumInputs(int inputNum)
    {
        float limit = std::sqrt(6.0f / inputNum);
        return getRandomFloat(-limit, limit);
    }
}