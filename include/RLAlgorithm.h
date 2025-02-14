#pragma once

#include "IEnvironment.h"

namespace BaseML
{
	class RLAlgorithm
	{
	private:
		IEnvironment environment;

	public:
		RLAlgorithm(const IEnvironment& environment) 
			:environment(environment) {}

		RLAlgorithm(IEnvironment&& environment) 
			:environment(std::move(environment)) {}

		virtual ~RLAlgorithm() = default; // Default destructor.

		// Learn the environment using the algorithm for 'maxIter' iterations.
		virtual void learn(size_t maxIter) = 0;
	};
}