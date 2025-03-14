#pragma once

#include <string>
#include <utility>
#include <list>

#include "Core/NeuralNetwork.h"
#include "Environment.h"
#include "RLAlgorithm.h"
#include "Utils/UtilsRandom.h"

namespace BaseML::RL
{
	class PPO : public RLAlgorithm
	{
	private:
		static constexpr size_t DEFAULT_HIDDEN_LAYER_SIZE = 64;

		std::string criticNetFile, actorNetFile;

		NeuralNetwork criticNetwork, actorNetwork;

		Utils::GaussianSampler sampler;

		float learningRate, rewardDiscountFactor, gaeLambda, clipThreshold, currEpisodeAvg, bestEpisodeAvg, saveThreshold;
		int timestepsPerBatch, maxTimestepsPerEpisode, minibatchSize, updatesPerIter;

		size_t timestepsLearned;

	public:
		PPO(std::shared_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, float learningRate = 0.005f,
			float discountFactor = 0.95f, float gaeLambda = 0.98f, int timestepsPerBatch = 4800, int maxTimestepsPerEpisode = 1600, 
			int minibatchSize = 400, int updatesPerIteration = 5, float actionSigma = 0.5f, float clipThreshold = 0.2f, float saveThreshold = -1.0f);

		// Change the starting learning rate of the algorithm
		void setLearningRate(float learningRate = 0.005f);

		// Set the standard deviation of the distribution from which the algorithm samples actions during training
		void setActionSigma(float actionSigma = 0.5f);

		// Set the size of the max episode length and the size of a single batch of data
		void setEpisodeAndBatchSize(int timestepsPerBatch = 4800, int maxTimestepsPerEpisode = 1600);

		// Set the size of each minibatch the algorithm trains on
		void setMinibatchSize(int minibatchSize = 800);

		// Set the number of times the algorithm should go over the same data
		void setUpdatesPerIter(int updatesPerIteration = 5);

		// Set the layer sizes of the critic network. 
		// Warning! This function deletes the old network parameters and resets the network's settings.
		void setCriticNetworkLayers(std::vector<size_t> layerSizes);

		// Set the layer sizes of the actor network. 
		// Warning! This function deletes the old network parameters and resets the network's settings.
		void setActorNetworkLayers(std::vector<size_t> layerSizes);

		// Set the output activation function of the actor network
		void setActorOutputActivationFunction(float (*activationFunction)(float),
			float (*activationFunctionDerivative)(float));

		// Set the save threshold for the networks (the amount of points to subtruct from the best score when 
		// choosing if the networks should be saved). Choose a negative value to always save the most recent 
		// networks.
		void setSaveThreshold(float saveThreshold = -1.0f);

		// Get a weak pointer to the environment of the algorithm
		std::weak_ptr<Environment> getEnvironment();

		// Loads the Networks from the files. Notice that the critic network file has additional data 
		// not related to the network. Returns true if successful and false if failed.
		bool loadFromFiles();

		void learn(size_t maxTimesteps) override;

		// Render and show the agent's performance in the environment in real time
		void showRealTime();

	private:
		// Get an action and its log probability from an observation. The first element in the returned pair 
		// is the action and the second is its log probability.
		std::pair<Matrix, float> getAction(const Matrix& observation);

		// Calculate Generalized Advantage Estimates for an episode based on the rewards and values of the episode. The calculated 
		// advantages will be appended in the right order to 'dest'.
		void computeGeneralizedAdvantageEstimates(std::list<float>& dest, const std::list<float>& rewards, const std::list<float>& values);

		// Convert a list containing single value numbers to a row vector represented by a Matrix
		Matrix scalarDataToMatrix(const std::list<float>& data);

		// Convert a list containing 1 dimensional vectors (represented as Matrix) to a 2 dimensional Matrix
		Matrix vectorDataToMatrix(const std::list<Matrix>& data);

		// Run the actor in the environment and collect data. Returns a pair of the data collected and the 
		// number of simulated timesteps.
		std::pair<RLTrainingData, size_t> collectTrajectories();

		// Get the current action means from the actor based on the observations and calculate the log probabilities 
		// of the current network choosing the given actions for the given observations. The first element in the returned 
		// pair is the action means and the second is the log probabilities.
		std::pair<Matrix, Matrix> checkActorUnderCurrentPolicy(const Matrix& observations, const Matrix& actions);

		// Calculate the gradients of the PPO-Clip objective and update the parameters of the actor network
		void updatePolicy(const RLTrainingData& data, float currentLearningRate);

		// Calculate the gradient of the mean-squared error to the real value of the states and update the 
		// parameters of the critic network
		void fitValueFunction(const RLTrainingData& data, float currentLearningRate);

		// Save Neural Networks to disk
		void save();

		// Generates a minibatch from the collected data using a pre-calculated randomly shaffled sequence
		RLTrainingData generateMinibatch(const RLTrainingData& data, const std::vector<int>& sequence, int start);
	};
}