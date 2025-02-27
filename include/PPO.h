#pragma once

#include <string>
#include <utility>

#include "NeuralNetwork.h"
#include "Environment.h"
#include "RLAlgorithm.h"
#include "UtilsRandom.h"

namespace BaseML::RL
{
	class PPO : public RLAlgorithm
	{
	private:
		static constexpr size_t DEFAULT_HIDDEN_LAYER_SIZE = 64;

		std::string criticNetFile, actorNetFile;

		NeuralNetwork criticNetwork, actorNetwork;

		Utils::GaussianSampler sampler;

		float learningRate, rewardDiscountFactor, clipThreshold;
		int timeStepsPerBatch, maxTimeStepsPerEpisode, updatesPerIter;

	public:
		PPO(std::unique_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, float learningRate = 0.005f,
			float discountFactor = 0.95f, float clipThreshold = 0.2f, int timeStepsPerBatch = 4800, int maxTimeStepsPerEpisode = 1600, 
			int updatesPerIteration = 5, float actionSigma = 0.3f);

		// Set the standard deviation of the distribution from which the algorithm samples actions during training
		void setActionSigma(float actionSigma);

		void learn(int maxTimeSteps) override;

	private:
		// Get an action and its log probability from an observation. The first element in the returned pair 
		// is the action and the second is its log probability.
		std::pair<const Matrix&, float> getAction(const Matrix& observation);

		// Calculate rewards-to-go for an episode based on the rewards from 'src'. The rtgs will be appended
		// in the right order to 'dest'.
		void calculateRewardsToGo(std::deque<float>& dest, const std::deque<float>& src);

		// Convert a deque containing single value numbers to a row vector represented by a Matrix
		Matrix scalarDataToMatrix(const std::deque<float>& data);

		// Convert a deque containing 1 dimensional vectors (represented as Matrix) to a 2 dimensional Matrix
		Matrix vectorDataToMatrix(const std::deque<Matrix>& data);

		// Run the actor in the environment and collect data
		RLTrainingData collectTrajectories();

		// Compute the estimated advantage using the critic network
		Matrix computeAdvantageEstimates(const RLTrainingData& data);

		// Calculate the log probabilities of the current network choosing the given actions for the given observations
		Matrix logProbabilitiesUnderCurrentPolicy(const Matrix& observations, const Matrix& actions);

		// Calculate the gradients of the PPO-Clip objective and update the parameters of the actor network
		void updatePolicy(const RLTrainingData& data, const Matrix& advantages);
	};
}