#include "PPO.h"

namespace BaseML::RL
{
	PPO::PPO(std::unique_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, float learningRate, float discountFactor,
		float clipThreshold, size_t timeStepsPerBatch, size_t maxTimeStepsPerEpisode, size_t updatesPerIteration)
		:RLAlgorithm(std::move(environment)), criticNetFile(criticFileName), actorNetFile(actorFileName), learningRate(learningRate), rewardDiscountFactor(discountFactor),
		clipThreshold(clipThreshold), timeStepsPerBatch(timeStepsPerBatch), maxTimeStepsPerEpisode(maxTimeStepsPerEpisode), updatesPerIter(updatesPerIteration), 
		criticNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, 1 }), 
		actorNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, this->environment->getActionDimension() })
	{
	}

	void PPO::learn(size_t maxTimeSteps)
	{
		size_t timeStepsPassed = 0; // Total time steps so far
		
		while (timeStepsPassed < maxTimeSteps)
		{

		}
	}

	RLTrainingData PPO::collectTrajectories()
	{
		RLTrainingData data;
	}
}