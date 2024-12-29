#include "layer.h"

#include <vector>
#include <cstdlib>

namespace MachineLearning
{
	Layer::Layer()
		:inputCount(0), outputCount(0), weights(), biases(), outputs()
	{
	}

	Layer::Layer(size_t numInputs, size_t numOutputs)
		:inputCount(numInputs), outputCount(numOutputs), weights(numOutputs, numInputs), 
		biases(numOutputs), outputs(numOutputs)
	{
		// Initialize the biases and weights with random values

		for (int i = 0; i < numOutputs; i++)
		{
			biases[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX_INIT_VAL)); // Generate a number between 0 and MAX_INIT_VAL (the number is a float)

			for (int j = 0; j < numInputs; j++)
			{
				weights(i, j) = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX_INIT_VAL));
			}
		}
	}

	void Layer::calculateOutputs(const std::vector<float>& inputs)
	{
		for (int i = 0; i < outputCount; i++)
		{
			outputs[i] = biases[i]; // Initialize output value with bias

			for (int j = 0; j < inputCount; j++)
			{
				outputs[i] += inputs[j] * weights(i, j);
			}
		}
	}

	size_t Layer::getInputCount()
	{
		return inputCount;
	}

	size_t Layer::getOutputCount()
	{
		return outputCount;
	}

	const std::vector<float>& Layer::getOutputs()
	{
		return outputs;
	}
}