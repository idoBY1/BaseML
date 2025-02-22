#pragma once

#include <string>
#include <utility>

#include "NeuralNetwork.h"
#include "Environment.h"
#include "RLAlgorithm.h"

namespace BaseML::RL
{
	class PPO : public RLAlgorithm
	{
	private:
		static constexpr size_t DEFAULT_HIDDEN_LAYER_SIZE = 64;

		std::string criticNetFile, actorNetFile;

		NeuralNetwork criticNetwork, actorNetwork;

		float learningRate, rewardDiscountFactor, clipThreshold;
		int timeStepsPerBatch, maxTimeStepsPerEpisode, updatesPerIter;

	public:
		PPO(std::unique_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, const char* playerId = NULL, float learningRate = 0.005f,
			float discountFactor = 0.95f, float clipThreshold = 0.2f, int timeStepsPerBatch = 4800, int maxTimeStepsPerEpisode = 1600, int updatesPerIteration = 5);

		void learn(int maxTimeSteps) override;

	private:
		//// Get an action and its log probability from an observation. The first element in the returned pair 
		//// is the action and the second is its log probability.
		//std::pair<const Matrix&, float> getAction(const Matrix& observation);

		//// Run the actor in the environment and collect data
		//RLTrainingData collectTrajectories();
	};
}