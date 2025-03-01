#include "PPO.h"

#include <deque>
#include <cmath>

#include "RLAlgorithm.h"
#include "UtilsGeneral.h"

namespace BaseML::RL
{
	PPO::PPO(std::unique_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, float learningRate, float discountFactor,
		float clipThreshold, int timestepsPerBatch, int maxTimestepsPerEpisode, int updatesPerIteration, float actionSigma)
		:RLAlgorithm(std::move(environment)), criticNetFile(criticFileName), actorNetFile(actorFileName), learningRate(learningRate), rewardDiscountFactor(discountFactor),
		clipThreshold(clipThreshold), timestepsPerBatch(timestepsPerBatch), maxTimestepsPerEpisode(maxTimestepsPerEpisode), updatesPerIter(updatesPerIteration), 
		criticNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, 1 }), 
		actorNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, this->environment->getActionDimension() }),
		sampler(actionSigma)
	{
		criticNetwork.setOutputActivationFunction([](float x) { return x; }, [](float x) { return 1.0f; });
		actorNetwork.setOutputActivationFunction([](float x) { return x; }, [](float x) { return 1.0f; });
	}

	void PPO::setActionSigma(float actionSigma)
	{
		sampler = Utils::GaussianSampler(actionSigma);
	}

	void PPO::learn(size_t maxTimesteps)
	{
		size_t timestepsPassed = 0; // Total time steps so far
		
		while (timestepsPassed < maxTimesteps)
		{
			auto [data, collectedTimesteps] = collectTrajectories();

			Matrix advantage = computeAdvantageEstimates(data);

			for (int i = 0; i < updatesPerIter; i++)
			{
				updatePolicy(data, advantage);

				fitValueFunction(data);
			}

			timestepsPassed += collectedTimesteps;
		}
	}

	void PPO::learnAndRender(size_t maxTimesteps)
	{
		size_t timestepsPassed = 0; // Total time steps so far

		while (timestepsPassed < maxTimesteps)
		{
			std::cout << "Started collecting data. current timestep: " << timestepsPassed << std::endl;
			auto [data, collectedTimesteps] = collectTrajectoriesRender();
			std::cout << "Finished collecting data. current timestep: " << timestepsPassed + collectedTimesteps << std::endl;

			Matrix advantage = computeAdvantageEstimates(data);

			for (int i = 0; i < updatesPerIter; i++)
			{
				updatePolicy(data, advantage);

				fitValueFunction(data);
			}

			timestepsPassed += collectedTimesteps;
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

	std::pair<RLTrainingData, size_t> PPO::collectTrajectories()
	{
		RLTrainingData data;

		int tBatch = 0, tEpisode;

		// Deques for collecting data (will be converted to objects of type Matrix)
		std::deque<Matrix> observations;
		std::deque<Matrix> actions;
		std::deque<float> logProbabilities;
		std::deque<float> rtgs; // Rewards-to-go

		while (tBatch < timestepsPerBatch)
		{
			environment->reset();

			std::deque<float> episodeRewards;

			for (tEpisode = 0; tEpisode < maxTimestepsPerEpisode && !environment->isFinished(); tEpisode++)
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
		}

		// Convert to matrices
		data.observations = vectorDataToMatrix(observations);
		data.actions = vectorDataToMatrix(actions);
		data.logProbabilities = scalarDataToMatrix(logProbabilities);
		data.rtgs = scalarDataToMatrix(rtgs);

		return { data, tBatch };
	}

	std::pair<RLTrainingData, size_t> PPO::collectTrajectoriesRender()
	{
		RLTrainingData data;

		int tBatch = 0, tEpisode;

		// Deques for collecting data (will be converted to objects of type Matrix)
		std::deque<Matrix> observations;
		std::deque<Matrix> actions;
		std::deque<float> logProbabilities;
		std::deque<float> rtgs; // Rewards-to-go

		float totalBatchReward = 0.0f; // for monitoring reward

		while (tBatch < timestepsPerBatch)
		{
			environment->reset();

			std::deque<float> episodeRewards;

			for (tEpisode = 0; tEpisode < maxTimestepsPerEpisode && !environment->isFinished(); tEpisode++)
			{
				tBatch++;

				// Get environment state
				const Matrix& observation = environment->getState(playerId.c_str());

				// Get action from actor network
				auto [action, logProbability] = getAction(observation);

				// Update environment
				environment->setAction(playerId.c_str(), action);
				environment->update(1.0f / 60.0f);

				// Render environment
				environment->render();

				// Get reward of action
				float reward = environment->getReward(playerId.c_str());

				totalBatchReward += reward;

				// Collect time step data
				observations.push_back(observation);
				actions.push_back(action);
				logProbabilities.push_back(logProbability);
				episodeRewards.push_back(reward);
			}

			// Compute rewards-to-go
			calculateRewardsToGo(rtgs, episodeRewards);
		}

		std::cout << "Total reward: " << totalBatchReward << std::endl;

		// Convert to matrices
		data.observations = vectorDataToMatrix(observations);
		data.actions = vectorDataToMatrix(actions);
		data.logProbabilities = scalarDataToMatrix(logProbabilities);
		data.rtgs = scalarDataToMatrix(rtgs);

		return { data, tBatch };
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

	std::pair<const Matrix&, const Matrix&> PPO::checkActorUnderCurrentPolicy(const Matrix& observations, const Matrix& actions)
	{
		Matrix actionMeans = actorNetwork.forwardPropagate(observations);

		return { actionMeans, sampler.batchLogProbabilities(actionMeans, actions) };
	}

	void PPO::updatePolicy(const RLTrainingData& data, const Matrix& advantages)
	{
		auto [currentActionMeans, currentLogProbabilities] = checkActorUnderCurrentPolicy(data.observations, data.actions);

		// Calculate the action-probability ratio of the current policy to the old policy
		Matrix ratios = currentLogProbabilities - data.logProbabilities;
		ratios.applyToElements([](float x) { return std::exp(x); });

		// Calculate gradients
		Matrix gradients(data.actions.rowsCount(), data.actions.columnsCount());

		// Iterate over all timesteps
		#pragma omp parallel for
		for (int i = 0; i < gradients.columnsCount(); i++)
		{
			// This 'if' statement calculates the gradients of the clip function and min function
			if (
				(advantages(i) > 0 && ratios(i) < 1 + clipThreshold) 
				|| 
				(advantages(i) < 0 && ratios(i) > 1 - clipThreshold)
				)
			{
				// Gradient of the advantage and probability ratio. The stddev is from the 
				// distribution's gradient calculation, but since it stays the same for all 
				// states we can divide by it here to save calculations. The minus (-) is 
				// here because we need to flip the sign of the gradient since we later use 
				// a gradient descent algorithm (instead of gradient ascent).
				float actionGradient = -advantages(i) * ratios(i) / (sampler.getSigma() * sampler.getSigma());

				for (int j = 0; j < gradients.rowsCount(); j++)
				{
					gradients(j, i) = actionGradient * (data.actions(j, i) - currentActionMeans(j, i));
				}
			}
			else // The clip is the smaller value and is outside of the clip range
			{
				// The gradient for this timestep should be 0
				for (int j = 0; j < gradients.rowsCount(); j++)
				{
					gradients(j, i) = 0.0f;
				}
			}
		}

		// Update actor network
		actorNetwork.backPropagation(gradients, learningRate);

		actorNetwork.saveToFile(actorNetFile.c_str());
	}

	void PPO::fitValueFunction(const RLTrainingData& data)
	{
		criticNetwork.learn(data.observations, data.rtgs, learningRate);

		criticNetwork.saveToFile(criticNetFile.c_str());
	}
}