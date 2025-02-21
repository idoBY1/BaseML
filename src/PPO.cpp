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

	std::pair<const Matrix&, float> PPO::getAction(const Matrix& observation)
	{
		// TODO: implement
	}

	RLTrainingData PPO::collectTrajectories()
	{
		RLTrainingData data;

		int tBatch = 0, tEpisode;

		while (tBatch < timeStepsPerBatch)
		{
			environment->reset();

			for (tEpisode = 0; tEpisode < maxTimeStepsPerEpisode && !environment->isFinished(); tEpisode++)
			{
				tBatch++;

				// Get environment state
				const Matrix& observation = environment->getState(playerId.c_str());

				// Get action from actor network
				auto [action, logProbability] = getAction(observation);

				// Update environment
				environment->setAction(playerId.c_str(), action);
				environment->update(1.0f / 60.0f);

				// Get reward of action
				float reward = environment->getReward(playerId.c_str());

				// Collect time step data
				data.observations.push_back(observation);
				data.actions.push_back(action);
				data.logProbabilities.push_back(logProbability);
				data.rewards.push_back(reward);
			}

			data.episodeLengths.push_back(tEpisode);
		}
	}
}