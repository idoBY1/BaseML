#pragma once

#include <random>

namespace BaseML::Utils
{
	class GaussianSampler
	{
	private:
		static constexpr float PI = 3.14159265359f;

		std::mt19937 gen; // RNG engine
		std::normal_distribution<float> distrib;

	public:
		// Create a new normal distribution with the given standard deviation
		GaussianSampler(float stddev);

		// Sample a value randomly from the normal distribution
		float sample(float mean);

		// Get the log probability density of a sample given the mean of the distibution
		float logProbabiltiy(float mean, float sample);
	};

	float getRandomFloat(float min, float max);
	float initFromNumInputs(int inputNum);
}