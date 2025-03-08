#include "PPO.h"

#include <deque>
#include <cmath>

#include "RLAlgorithm.h"
#include "UtilsGeneral.h"

namespace BaseML::RL
{
	PPO::PPO(std::shared_ptr<Environment> environment, const char* criticFileName, const char* actorFileName, float learningRate, float discountFactor, float gaeLambda, 
		int timestepsPerBatch, int maxTimestepsPerEpisode, int minibatchSize, int updatesPerIteration, float actionSigma, float clipThreshold)
		:RLAlgorithm(environment), criticNetFile(criticFileName), actorNetFile(actorFileName), learningRate(learningRate), 
		rewardDiscountFactor(discountFactor), gaeLambda(gaeLambda), timestepsPerBatch(timestepsPerBatch),
		maxTimestepsPerEpisode(maxTimestepsPerEpisode), minibatchSize(minibatchSize), clipThreshold(clipThreshold), updatesPerIter(updatesPerIteration), 
		criticNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, 1 }), 
		actorNetwork({ this->environment->getObservationDimension(), DEFAULT_HIDDEN_LAYER_SIZE, this->environment->getActionDimension() }),
		sampler(actionSigma), timestepsLearned(0)
	{
		criticNetwork.setOutputActivationFunction([](float x) { return x; }, [](float x) { return 1.0f; });
		actorNetwork.setOutputActivationFunction([](float x) { return x; }, [](float x) { return 1.0f; });
	}

	void PPO::setLearningRate(float learningRate)
	{
		this->learningRate = learningRate;
	}

	void PPO::setActionSigma(float actionSigma)
	{
		sampler = Utils::GaussianSampler(actionSigma);
	}

	void PPO::setEpisodeAndBatchSize(int timestepsPerBatch, int maxTimestepsPerEpisode)
	{
		this->timestepsPerBatch = timestepsPerBatch;
		this->maxTimestepsPerEpisode = maxTimestepsPerEpisode;
	}

	void PPO::setMinibatchSize(int minibatchSize)
	{
		this->minibatchSize = minibatchSize;
	}

	void PPO::setUpdatesPerIter(int updatesPerIteration)
	{
		this->updatesPerIter = updatesPerIteration;
	}

	void PPO::setCriticNetworkLayers(std::vector<size_t> layerSizes)
	{
		layerSizes.insert(layerSizes.begin(), this->environment->getObservationDimension());
		layerSizes.push_back(1);

		criticNetwork = NeuralNetwork(layerSizes);
	}

	void PPO::setActorNetworkLayers(std::vector<size_t> layerSizes)
	{
		layerSizes.insert(layerSizes.begin(), this->environment->getObservationDimension());
		layerSizes.push_back(this->environment->getActionDimension());

		actorNetwork = NeuralNetwork(layerSizes);
	}

	void PPO::setActorOutputActivationFunction(float(*activationFunction)(float), float(*activationFunctionDerivative)(float))
	{
		actorNetwork.setOutputActivationFunction(activationFunction, activationFunctionDerivative);
	}

	std::weak_ptr<Environment> PPO::getEnvironment()
	{
		return environment;
	}

	bool PPO::loadFromFiles()
	{
		if (!actorNetwork.loadFromFile(actorNetFile.c_str()))
			return false;

		actorNetwork.setOutputActivationFunction([](float x) { return x; }, [](float x) { return 1.0f; });

		try {
			std::ifstream ifile;

			ifile.open(criticNetFile, std::ios::binary | std::ios::in);

			ifile.read(reinterpret_cast<char*>(&timestepsLearned), sizeof(timestepsLearned));
			criticNetwork.load(ifile);

			ifile.close();
		}
		catch (...) {
			return false;
		}

		criticNetwork.setOutputActivationFunction([](float x) { return x; }, [](float x) { return 1.0f; });

		std::cout << "Continuing from step: " << timestepsLearned << std::endl;

		return true;
	}

	void PPO::learn(size_t maxTimesteps)
	{
		size_t timestepsPassed = 0; // Total time steps so far
		
		while (timestepsPassed < maxTimesteps)
		{
			auto [data, collectedTimesteps] = collectTrajectories();

			for (int i = 0; i < updatesPerIter; i++)
			{
				int timestepsUsedFromBatch = 0;
				std::vector<int> shuffledIndexes = Utils::generateShuffledNumberSequence(collectedTimesteps);

				while (timestepsUsedFromBatch < collectedTimesteps)
				{
					RLTrainingData minibatch = generateMinibatch(data, shuffledIndexes, timestepsUsedFromBatch);

					// Calculate learning rate for current step
					float currentLearningRate = learningRate * (1.0f - ((timestepsPassed + timestepsUsedFromBatch) / maxTimesteps));

					// Update networks
					updatePolicy(minibatch, currentLearningRate);
					fitValueFunction(minibatch, currentLearningRate);

					timestepsUsedFromBatch += minibatchSize;
				}
			}

			timestepsPassed += collectedTimesteps;
			timestepsLearned += collectedTimesteps;

			save();
		}
	}

	void PPO::showRealTime()
	{
		float episodeReward = 0.0f;
		int totalTimesteps = 0;

		environment->close();
		environment->initialize(true);

		environment->reset();

		while (!environment->isFinished() && totalTimesteps < maxTimestepsPerEpisode)
		{
			totalTimesteps++;

			// Get environment state
			const Matrix& observation = environment->getState(playerId.c_str());

			// Get action from actor network
			Matrix action = actorNetwork.forwardPropagate(observation);

			// Update environment
			environment->setAction(playerId.c_str(), action);
			environment->update();

			// Render environment
			environment->render();

			// Get reward of action
			float reward = environment->getReward(playerId.c_str());

			episodeReward += reward;
		}

		std::cout << "Total episode reward: " << episodeReward << std::endl;

		environment->close();
		environment->initialize(false);
	}

	std::pair<Matrix, float> PPO::getAction(const Matrix& observation)
	{
		Matrix actionMean = actorNetwork.forwardPropagate(observation);

		Matrix action = sampler.sample(actionMean);
		float logProbability = sampler.logProbabiltiy(actionMean, action);

		return { action, logProbability };
	}

	void PPO::computeGeneralizedAdvantageEstimates(std::deque<float>& dest, const std::deque<float>& rewards, const std::deque<float>& values)
	{
		std::deque<float> temp;

		float delta, gae;

		gae = rewards[rewards.size() - 1] - values[values.size() - 1];

		temp.push_front(gae); // The last GAE doesn't have future values to account for

		for (int i = rewards.size() - 2; i >= 0; i--)
		{
			delta = rewards[i] + rewardDiscountFactor * values[i + 1] - values[i];
			gae = delta + rewardDiscountFactor * gaeLambda * gae;
			temp.push_front(gae);
		}

		// Append the GAEs to the queue of the batch
		for (int i = 0; i < temp.size(); i++)
		{
			dest.push_back(temp[i]);
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
		std::deque<float> advantages;
		std::deque<float> stateValues;

		// For monitoring reward
		float totalBatchReward = 0.0f; 
		int numEpisodes = 0;

		while (tBatch < timestepsPerBatch)
		{
			environment->reset();

			std::deque<float> episodeRewards;
			std::deque<float> episodeValues;
			numEpisodes++;

			// Run an episode
			for (tEpisode = 0; tEpisode < maxTimestepsPerEpisode && !environment->isFinished(); tEpisode++)
			{
				tBatch++;

				// Get environment state
				const Matrix& observation = environment->getState(playerId.c_str());

				// Get action from actor network
				auto [action, logProbability] = getAction(observation);

				// Get value estimate from critic network
				float valueEstimate = criticNetwork.forwardPropagate(observation)(0);
				episodeValues.push_back(valueEstimate);

				// Update environment
				environment->setAction(playerId.c_str(), action);
				environment->update();

				// Get reward of action
				float reward = environment->getReward(playerId.c_str());
				episodeRewards.push_back(reward);

				totalBatchReward += reward;

				// Collect time step data
				observations.push_back(observation);
				actions.push_back(action);
				logProbabilities.push_back(logProbability);
				stateValues.push_back(valueEstimate);
			}

			// Compute GAE
			computeGeneralizedAdvantageEstimates(advantages, episodeRewards, episodeValues);
		}

		std::cout << "Average episode reward: " << totalBatchReward / (float)numEpisodes << std::endl;

		// Convert to matrices
		data.observations = vectorDataToMatrix(observations);
		data.actions = vectorDataToMatrix(actions);
		data.logProbabilities = scalarDataToMatrix(logProbabilities);
		data.advantages = scalarDataToMatrix(advantages);
		data.stateValues = scalarDataToMatrix(stateValues);

		return { data, tBatch };
	}

	std::pair<Matrix, Matrix> PPO::checkActorUnderCurrentPolicy(const Matrix& observations, const Matrix& actions)
	{
		Matrix actionMeans = actorNetwork.forwardPropagate(observations);

		return { actionMeans, sampler.batchLogProbabilities(actionMeans, actions) };
	}

	void PPO::updatePolicy(const RLTrainingData& data, float currentLearningRate)
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
				(data.advantages(i) > 0 && ratios(i) < 1 + clipThreshold) 
				||
				(data.advantages(i) < 0 && ratios(i) > 1 - clipThreshold)
				)
			{
				// Gradient of the advantage and probability ratio. The stddev is from the 
				// distribution's gradient calculation, but since it stays the same for all 
				// states we can divide by it here to save calculations. The minus (-) is 
				// here because we need to flip the sign of the gradient since we later use 
				// a gradient descent algorithm (instead of gradient ascent).
				float actionGradient = -data.advantages(i) * ratios(i) / (sampler.getSigma() * sampler.getSigma());

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
		actorNetwork.backPropagation(gradients, currentLearningRate);
	}

	void PPO::fitValueFunction(const RLTrainingData& data, float currentLearningRate)
	{
		criticNetwork.learn(data.observations, (data.advantages + data.stateValues), currentLearningRate);
	}

	void PPO::save()
	{
		actorNetwork.saveToFile(actorNetFile.c_str());

		std::ofstream ofile;

		ofile.open(criticNetFile.c_str(), std::ios::binary | std::ios::out);

		ofile.write(reinterpret_cast<const char*>(&timestepsLearned), sizeof(timestepsLearned));
		criticNetwork.save(ofile);

		ofile.close();
	}

	RLTrainingData PPO::generateMinibatch(const RLTrainingData& data, const std::vector<int>& sequence, int start)
	{
		RLTrainingData minibatch;
		int actualMinibatchSize = (minibatchSize < (sequence.size() - start)) ? minibatchSize : (sequence.size() - start);

		minibatch.observations = Matrix(environment->getObservationDimension(), actualMinibatchSize);
		minibatch.actions = Matrix(environment->getActionDimension(), actualMinibatchSize);
		minibatch.logProbabilities = Matrix(1, actualMinibatchSize);
		minibatch.advantages = Matrix(1, actualMinibatchSize);
		minibatch.stateValues = Matrix(1, actualMinibatchSize);

		for (int step = 0; step < actualMinibatchSize; step++)
		{
			int shuffledTimestep = sequence[start + step];

			// Copy observations
			for (int obsIndx = 0; obsIndx < environment->getObservationDimension(); obsIndx++)
			{
				minibatch.observations(obsIndx, step) = data.observations(obsIndx, shuffledTimestep);
			}

			// Copy actions
			for (int actIndx = 0; actIndx < environment->getActionDimension(); actIndx++)
			{
				minibatch.actions(actIndx, step) = data.actions(actIndx, shuffledTimestep);
			}

			// Copy log probabilities
			minibatch.logProbabilities(step) = data.logProbabilities(shuffledTimestep);

			// Copy advantages
			minibatch.advantages(step) = data.advantages(shuffledTimestep);

			// Copy state values
			minibatch.stateValues(step) = data.stateValues(shuffledTimestep);
		}

		return minibatch;
	}
}