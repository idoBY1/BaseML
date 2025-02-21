#include "PPO.h"

namespace BaseML::RL
{
	PPO::PPO(std::unique_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, const char* playerId, float learningRate, float discountFactor,
		float clipThreshold, int timeStepsPerBatch, int maxTimeStepsPerEpisode, int updatesPerIteration)
		:RLAlgorithm(std::move(environment), playerId), criticNetFile(criticFileName), actorNetFile(actorFileName), learningRate(learningRate), rewardDiscountFactor(discountFactor),
		clipThreshold(clipThreshold), timeStepsPerBatch(timeStepsPerBatch), maxTimeStepsPerEpisode(maxTimeStepsPerEpisode), updatesPerIter(updatesPerIteration), 
		criticNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, 1 }), 
		actorNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, this->environment->getActionDimension() })
	{
	}

	void PPO::learn(int maxTimeSteps)
	{
		int timeStepsPassed = 0; // Total time steps so far
		
		while (timeStepsPassed < maxTimeSteps)
		{

		}
	}

	RLTrainingData PPO::collectTrajectories()
	{
		RLTrainingData data;
		bool episodeFinished;

		int tBatch = 0;

		while (tBatch < timeStepsPerBatch)
		{
			environment->reset();
			episodeFinished = false;

			for (int i = 0; i < maxTimeStepsPerEpisode; i++)
			{
				tBatch++;

				data.observations.push_back(environment->getState(playerId.c_str()));
			}
		}
	}
}