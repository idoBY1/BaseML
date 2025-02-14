#pragma once

#include "IEnvironment.h"

namespace BaseML
{
	class RLAlgorithm
	{
	protected:
		IEnvironment* environment; // switch to unique_ptr

	public:
		// Create a new RLAlgorithm. Clones 'environment'.
		RLAlgorithm(const IEnvironment& environment) 
			:environment(environment.clone()) 
		{}

		// Create a new RLAlgorithm. Assumes ownership on 'environment'.
		RLAlgorithm(IEnvironment* environment) 
			:environment(environment) 
		{}

		virtual ~RLAlgorithm() {
			delete environment; // Free environment when destroyed.
		}

		// Learn the environment using the algorithm for 'maxIter' iterations.
		virtual void learn(size_t maxIter) = 0;
	};
}