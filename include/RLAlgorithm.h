#pragma once

#include "Environment.h"

namespace BaseML
{
	class RLAlgorithm
	{
	protected:
		std::unique_ptr<Environment> environment;

	public:
		// Create a new RLAlgorithm. Takes ownership on 'environment'.
		RLAlgorithm(std::unique_ptr<Environment> environment) 
			:environment(std::move(environment)) 
		{}

		virtual ~RLAlgorithm() = default;

		// Learn the environment using the algorithm for 'maxIter' iterations.
		virtual void learn(size_t maxIter) = 0;
	};
}