#pragma once

#include <random>
#include <vector>

#include "Matrix.h"

namespace BaseML::Utils
{
	class GaussianSampler
	{
	private:
		static constexpr float SAMPLER_PI = 3.14159265359f; // Called SAMPLER_PI and not PI to avoid name conflicts

		std::mt19937 gen; // RNG engine
		std::normal_distribution<float> distrib;

	public:
		// Create a new normal distribution with the given standard deviation
		GaussianSampler(float stddev);

		// Get the standard deviation of the sampler
		float getSigma();

		// Sample a value randomly from the normal distribution
		float sample(float mean);

		// Sample a vector of values randomly from the normal distribution
		Matrix sample(const Matrix& mean);

		// Get the log probability density of a sample given the mean of the distibution
		float logProbabiltiy(float mean, float sample);

		// Get the log probability density of a sample given the mean of the distibution
		float logProbabiltiy(const Matrix& mean, const Matrix& sample);

		// Get the log probability density of samples given the means of the distibutions. This function 
		// assumes that each mean and sample is a column in the matrices and that the standard deviation 
		// of the distributions is the same. The function returns a row vector (as a Matrix) of the 
		// log-probabilities of the actions.
		Matrix batchLogProbabilities(const Matrix& means, const Matrix& samples);
	};

	// Returns a randomly generated floating point number
	float getRandomFloat(float min, float max);

	// Given the number of inputs, generates a random starting value for a network parameter in the appropriate range 
	float initFromNumInputs(int inputNum);

	// Generate a sequence of numbers from 0 to 'rangeLength'-1 and shuffle them randomly
	std::vector<int> generateShuffledNumberSequence(int rangeLength);
}