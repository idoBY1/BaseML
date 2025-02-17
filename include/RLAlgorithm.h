#pragma once

#include "IEnvironment.h"

namespace BaseML
{
	class RLAlgorithm
	{
	protected:
		std::unique_ptr<IEnvironment> environment;

	public:
		// Create a new RLAlgorithm. Clones 'environment'.
		RLAlgorithm(const IEnvironment& environment) 
			:environment(environment.clone()) 
		{}

		// Create a new RLAlgorithm. Assumes ownership on 'environment'.
		RLAlgorithm(std::unique_ptr<IEnvironment> environment) 
			:environment(std::move(environment)) 
		{}

		// Learn the environment using the algorithm for 'maxIter' iterations.
		virtual void learn(size_t maxIter) = 0;
	};
}