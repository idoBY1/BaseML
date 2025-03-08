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
		std::shared_ptr<Environment> environment;
		std::string playerId;

	public:
		// Create a new RLAlgorithm.
		RLAlgorithm(std::shared_ptr<Environment> environment, const char* playerId = NULL)
			:environment(environment) 
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

		void setPlayerId(const char* id)
		{
			playerId = id;
		}

		// Learn the environment using the algorithm for 'maxIter' iterations.
		virtual void learn(size_t maxIter) = 0;
	};

	// Training data to be used by classes implementing RLAlgorithm
	struct RLTrainingData
	{
		Matrix observations;
		Matrix actions;
		Matrix logProbabilities;
		Matrix advantages;
		Matrix stateValues;
	};
}