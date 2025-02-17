#pragma once

#include "NeuralNetwork.h"
#include "RLAlgorithm.h"

namespace BaseML
{
	class PPO : public RLAlgorithm
	{
	private:
		std::string neuralNetFile;

		NeuralNetwork neuralNet;

		float learningRate, rewardDiscountFactor, clipThreshold;
		size_t timeStepsPerBatch, maxTimeStepsPerEpisode, updatesPerIter;

	public:
		PPO(const IEnvironment& environment, const char* neuralNetworkFileName, float learningRate = 0.005f, float discountFactor = 0.95f, float clipThreshold = 0.2f, 
			size_t timeStepsPerBatch = 4800, size_t maxTimeStepsPerEpisode = 1600, size_t updatesPerIteration = 5);

		PPO(std::unique_ptr<IEnvironment> environment, const char* neuralNetworkFileName, float learningRate = 0.005f, float discountFactor = 0.95f, float clipThreshold = 0.2f,
			size_t timeStepsPerBatch = 4800, size_t maxTimeStepsPerEpisode = 1600, size_t updatesPerIteration = 5);

		void learn(size_t maxIter) override;
	};
}