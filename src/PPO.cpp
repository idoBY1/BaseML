#include "PPO.h"

#include <deque>

namespace BaseML::RL
{
	PPO::PPO(std::unique_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, float learningRate, float discountFactor,
		float clipThreshold, int timeStepsPerBatch, int maxTimeStepsPerEpisode, int updatesPerIteration, float actionSigma)
		:RLAlgorithm(std::move(environment)), criticNetFile(criticFileName), actorNetFile(actorFileName), learningRate(learningRate), rewardDiscountFactor(discountFactor),
		clipThreshold(clipThreshold), timeStepsPerBatch(timeStepsPerBatch), maxTimeStepsPerEpisode(maxTimeStepsPerEpisode), updatesPerIter(updatesPerIteration), 
		criticNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, 1 }), 
		actorNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, this->environment->getActionDimension() }),
		sampler(actionSigma)
	{
	}

	void PPO::setActionSigma(float actionSigma)
	{
		sampler = Utils::GaussianSampler(actionSigma);
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
		Matrix actionMean = actorNetwork.forwardPropagate(observation);

		Matrix action = sampler.sample(actionMean);
		float logProbability = sampler.logProbabiltiy(actionMean, action);

		return { action, logProbability };
	}

	void PPO::calculateRewardsToGo(std::deque<float>& dest, const std::deque<float>& src)
	{
		std::deque<float> temp;
		float discountedReward = 0.0f;

		// Start computing from last reward to compute rtgs correctly
		for (auto r = src.end(); r != src.begin(); r--)
		{
			discountedReward = *r + discountedReward * rewardDiscountFactor;
			temp.push_front(discountedReward); // Insert in reverse order
		}

		// Append the rtgs to the queue of the batch
		for (auto rtg = src.begin(); rtg != src.end(); rtg++)
		{
			dest.push_back(*rtg);
		}
	}

	RLTrainingData PPO::collectTrajectories()
	{
		RLTrainingData data;

		int tBatch = 0, tEpisode;

		while (tBatch < timeStepsPerBatch)
		{
			environment->reset();

			std::deque<float> episodeRewards;

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
				episodeRewards.push_back(reward);
			}

			// Compute rewards-to-go
			calculateRewardsToGo(data.rtgs, episodeRewards);

			// Collect the length of this episode
			data.episodeLengths.push_back(tEpisode);
		}

		return data;
	}
}