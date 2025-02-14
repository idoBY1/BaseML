#pragma once

#include "RLAlgorithm.h"

namespace BaseML
{
	class PPO : public RLAlgorithm
	{
	private:
		std::string neuralNetFile;

		float learningRate, rewardDiscountFactor, clipThreshold;
		size_t timeStepsPerBatch, maxTimeStepsPerEpisode, updatesPerIter;

	public:
		PPO(); // TODO: initialize fields in constructor

		// TODO: implement RLAlgorithm methods
	};
}