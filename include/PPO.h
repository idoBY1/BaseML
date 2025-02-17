#pragma once

#include "NeuralNetwork.h"
#include "Environment.h"
#include "RLAlgorithm.h"

namespace BaseML
{
	class PPO : public RLAlgorithm
	{
	private:
		static constexpr size_t DEFAULT_HIDDEN_LAYER_SIZE = 64;

		std::string criticNetFile, actorNetFile;

		NeuralNetwork criticNetwork, actorNetwork;

		float learningRate, rewardDiscountFactor, clipThreshold;
		size_t timeStepsPerBatch, maxTimeStepsPerEpisode, updatesPerIter;

	public:
		PPO(std::unique_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, float learningRate = 0.005f,
			float discountFactor = 0.95f, float clipThreshold = 0.2f, size_t timeStepsPerBatch = 4800, size_t maxTimeStepsPerEpisode = 1600, size_t updatesPerIteration = 5);

		void learn(size_t maxIter) override;
	};
}