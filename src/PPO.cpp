#include "PPO.h"

namespace BaseML
{
	PPO::PPO(const IEnvironment& environment, const char* neuralNetworkFileName, float learningRate, float discountFactor, 
		float clipThreshold, size_t timeStepsPerBatch, size_t maxTimeStepsPerEpisode, size_t updatesPerIteration)
		:RLAlgorithm(environment), neuralNetFile(neuralNetworkFileName), learningRate(learningRate), rewardDiscountFactor(discountFactor), 
		clipThreshold(clipThreshold), timeStepsPerBatch(timeStepsPerBatch), maxTimeStepsPerEpisode(maxTimeStepsPerEpisode), updatesPerIter(updatesPerIteration)
	{
	}

	PPO::PPO(std::unique_ptr<IEnvironment> environment, const char* neuralNetworkFileName, float learningRate, float discountFactor, 
		float clipThreshold, size_t timeStepsPerBatch, size_t maxTimeStepsPerEpisode, size_t updatesPerIteration)
		:RLAlgorithm(std::move(environment)), neuralNetFile(neuralNetworkFileName), learningRate(learningRate), rewardDiscountFactor(discountFactor),
		clipThreshold(clipThreshold), timeStepsPerBatch(timeStepsPerBatch), maxTimeStepsPerEpisode(maxTimeStepsPerEpisode), updatesPerIter(updatesPerIteration)
	{
	}

	void PPO::learn(size_t maxIter)
	{
	}
}