#pragma once

#include <string>
#include <deque>

#include "Matrix.h"
#include "Environment.h"

namespace BaseML::RL
{
	class RLAlgorithm
	{
	protected:
		std::unique_ptr<Environment> environment;
		std::string playerId;

	public:
		// Create a new RLAlgorithm. Takes ownership on 'environment'.
		RLAlgorithm(std::unique_ptr<Environment> environment, const char* playerId = NULL)
			:environment(std::move(environment)) 
		{
			if (!this->environment->isInitialized())
				this->environment->initialize();

			if (playerId)
				this->playerId = playerId;
			else
				this->playerId = this->environment->getPlayers().at(0);
		}

		virtual ~RLAlgorithm()
		{
			if (environment->isInitialized())
				environment->close();
		}

		// Learn the environment using the algorithm for 'maxIter' iterations.
		virtual void learn(int maxIter) = 0;
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