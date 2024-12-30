#include "layer.h"

#include <vector>
#include <cstdlib>
#include <cmath>

#include "utils.h"

namespace MachineLearning
{
	Layer::Layer()
		:inputCount(0), outputCount(0), weights(), biases(), outputs(), gradients()
	{
	}

	Layer::Layer(size_t numInputs, size_t numOutputs)
		:inputCount(numInputs), outputCount(numOutputs), weights(numOutputs, numInputs), 
		biases(numOutputs), outputs(numOutputs), gradients(numOutputs)
	{
		// Initialize the biases and weights with random values

		for (int i = 0; i < numOutputs; i++)
		{
			// Generate a number between MIN_INIT_VAL and MAX_INIT_VAL (the number is a float)
			biases[i] = Utils::getRandomFloat(MIN_INIT_VAL, MAX_INIT_VAL);

			for (int j = 0; j < numInputs; j++)
			{
				weights(i, j) = Utils::getRandomFloat(MIN_INIT_VAL, MAX_INIT_VAL);
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

	// The activation function
	float Layer::sigmoid(float input)
	{
		return 1.0f / (1.0f + expf(-input));
	}

	// The derivative of the activation function (the derivative of the sigmoid 
	// function can be calculated from its output)
	float Layer::sigmoidDerivative(float neuronOutput)
	{
		return neuronOutput * (1.0f - neuronOutput);
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

			outputs[i] = sigmoid(outputs[i]);
		}
	}
}