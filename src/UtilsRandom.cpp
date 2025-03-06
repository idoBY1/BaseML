#include "UtilsRandom.h"

#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace BaseML::Utils
{
    GaussianSampler::GaussianSampler(float stddev)
        :gen(std::random_device{}()), distrib(0.0f, stddev)
    {
    }

    float GaussianSampler::getSigma()
    {
        return distrib.stddev();
    }

    float GaussianSampler::sample(float mean)
    {
        return mean + distrib(gen);
    }

    Matrix GaussianSampler::sample(const Matrix& mean)
    {
        Matrix sample(mean);

        #pragma omp parallel for
        for (int i = 0; i < sample.size(); i++)
        {
            sample(i) += distrib(gen);
        }

        return sample;
    }

    float GaussianSampler::logProbabiltiy(float mean, float sample)
    {
        float diff = sample - mean;
        return -std::log(distrib.stddev() * std::sqrt(2.0f * SAMPLER_PI)) - 0.5f * (diff * diff) / (distrib.stddev() * distrib.stddev());
    }

    float GaussianSampler::logProbabiltiy(const Matrix& mean, const Matrix& sample)
    {
#ifdef DEBUG
        if (mean.size() != sample.size())
        {
            std::cout << "Invalid sizes in log probability calculation" << std::endl;
            throw std::runtime_error("Invalid sizes in log probability");
        }
#endif // DEBUG

        float logProbability = 0.0f;

        float sharedPart = std::log(distrib.stddev() * std::sqrt(2.0f * SAMPLER_PI));

        for (int i = 0; i < sample.size(); i++)
        {
            float diff = sample(i) - mean(i);
            logProbability += -sharedPart - 0.5f * (diff * diff) / (distrib.stddev() * distrib.stddev());
        }

        return logProbability;
    }

    Matrix GaussianSampler::batchLogProbabilities(const Matrix& means, const Matrix& samples)
    {
#ifdef DEBUG
        if (means.rowsCount() != samples.rowsCount() || means.columnsCount() != samples.columnsCount())
        {
            std::cout << "Invalid sizes in log probability calculation" << std::endl;
            throw std::runtime_error("Invalid sizes in log probability");
        }
#endif // DEBUG

        Matrix logProbs(1, samples.columnsCount());

        float sharedPart = std::log(distrib.stddev() * std::sqrt(2.0f * SAMPLER_PI));

        for (int i = 0; i < samples.columnsCount(); i++)
        {
            float logProbability = 0.0f;

            for (int j = 0; j < samples.rowsCount(); j++)
            {
                float diff = samples(j, i) - means(j, i);
                logProbability += -sharedPart - 0.5f * (diff * diff) / (distrib.stddev() * distrib.stddev());
            }

            logProbs(i) = logProbability;
        }

        return logProbs;
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

    std::vector<int> generateShuffledNumberSequence(int rangeLength)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::vector<int> values(rangeLength);

        std::iota(values.begin(), values.end(), 0); // Fill the vector with a sequence of values

        std::shuffle(values.begin(), values.end(), gen);

        return values;
    }
}