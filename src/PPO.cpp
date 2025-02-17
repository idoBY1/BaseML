#include "PPO.h"

namespace BaseML
{
	PPO::PPO(std::unique_ptr<IEnvironment> environment, std::unique_ptr<NeuralNetwork> neuralNetwork, const char* neuralNetworkFileName, float learningRate, float discountFactor,
		float clipThreshold, size_t timeStepsPerBatch, size_t maxTimeStepsPerEpisode, size_t updatesPerIteration)
		:RLAlgorithm(std::move(environment)), neuralNet(std::move(neuralNetwork)), neuralNetFile(neuralNetworkFileName), learningRate(learningRate), rewardDiscountFactor(discountFactor),
		clipThreshold(clipThreshold), timeStepsPerBatch(timeStepsPerBatch), maxTimeStepsPerEpisode(maxTimeStepsPerEpisode), updatesPerIter(updatesPerIteration)
	{
	}

	void PPO::learn(size_t maxIter)
	{
	}
}