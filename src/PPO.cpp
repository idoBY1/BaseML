#include "PPO.h"

#include <deque>

#include "RLAlgorithm.h"
#include "UtilsGeneral.h"

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
			RLTrainingData data = collectTrajectories();

			Matrix advantage = computeAdvantageEstimates(data);
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

	Matrix PPO::scalarDataToMatrix(const std::deque<float>& data)
	{
		Matrix converted(1, data.size());

		for (int i = 0; i < data.size(); i++)
		{
			converted(i) = data[i];
		}

		return converted;
	}

	Matrix PPO::vectorDataToMatrix(const std::deque<Matrix>& data)
	{
		if (data.size() == 0)
		{
			std::cout << "Cannot convert an empty collection" << std::endl;
			throw std::runtime_error("Cannot convert an empty collection");
		}

		Matrix converted(data[0].size(), data.size());

		for (int i = 0; i < converted.columnsCount(); i++)
		{
			for (int j = 0; j < converted.rowsCount(); j++)
			{
				converted(j, i) = data[i](j);
			}
		}

		return converted;
	}

	RLTrainingData PPO::collectTrajectories()
	{
		RLTrainingData data;

		int tBatch = 0, tEpisode;

		// Deques for collecting data (will be converted to objects of type Matrix)
		std::deque<Matrix> observations;
		std::deque<Matrix> actions;
		std::deque<float> logProbabilities;
		std::deque<float> rtgs; // Rewards-to-go

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
				observations.push_back(observation);
				actions.push_back(action);
				logProbabilities.push_back(logProbability);
				episodeRewards.push_back(reward);
			}

			// Compute rewards-to-go
			calculateRewardsToGo(rtgs, episodeRewards);

			// Collect the length of this episode
			data.episodeLengths.push_back(tEpisode);
		}

		// Convert to matrices
		data.observations = vectorDataToMatrix(observations);
		data.actions = vectorDataToMatrix(actions);
		data.logProbabilities = scalarDataToMatrix(logProbabilities);
		data.rtgs = scalarDataToMatrix(rtgs);

		return data;
	}

	Matrix PPO::computeAdvantageEstimates(const RLTrainingData& data)
	{
		Matrix criticStateValues = criticNetwork.forwardPropagate(data.observations);

		// Calculate advantages
		Matrix advantages = data.rtgs - criticStateValues;

		// Normalize advanteges for numerical stability
		advantages = Utils::zScoreNormalize(advantages);

		return advantages;
	}
}