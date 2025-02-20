#pragma once

#include <deque>

#include "Matrix.h"
#include "Environment.h"

namespace BaseML::RL
{
	class RLAlgorithm
	{
	protected:
		std::unique_ptr<Environment> environment;

	public:
		// Create a new RLAlgorithm. Takes ownership on 'environment'.
		RLAlgorithm(std::unique_ptr<Environment> environment) 
			:environment(std::move(environment)) 
		{
			if (!environment->isInitialized())
				environment->initialize();
		}

		virtual ~RLAlgorithm()
		{
			if (environment->isInitialized())
				environment->close();
		}

		// Learn the environment using the algorithm for 'maxIter' iterations.
		virtual void learn(size_t maxIter) = 0;
	};

	// Training data to be used by classes implementing RLAlgorithm
	struct RLTrainingData
	{
		std::deque<Matrix> observations;
		std::deque<Matrix> actions;
		std::deque<float> logProbabilities;
		std::deque<float> rewards;
		std::deque<float> rtgs; // Reward-to-gos
		std::deque<size_t> episodeLengths;
	};
}