#include "layer.h"

#include <vector>
#include <cstdlib>
#include <cmath>

#include "utils.h"

namespace MachineLearning
{
	Layer::Layer()
		:inputCount(0), outputCount(0), weights(), biases(), outputs(), gradients(), activationFunc(nullptr), activationFuncDerivative(nullptr), lossFunc(nullptr)
	{
	}

	Layer::Layer(size_t numInputs, size_t numOutputs)
		:inputCount(numInputs), outputCount(numOutputs), weights(numOutputs, numInputs), biases(numOutputs), outputs(numOutputs), 
		gradients(numOutputs), activationFunc(&Utils::sigmoid), activationFuncDerivative(&Utils::sigmoidDerivative), lossFunc(&Utils::squareError)
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

	Layer::Layer(size_t numInputs, size_t numOutputs, float(*activationFunction)(float), 
		float(*activationFunctionDerivative)(float), float (*lossFunction)(float, float))
		:inputCount(numInputs), outputCount(numOutputs), weights(numOutputs, numInputs), biases(numOutputs), outputs(numOutputs), gradients(numOutputs), 
		activationFunc(activationFunction), activationFuncDerivative(activationFunctionDerivative), lossFunc(lossFunction)
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

	size_t Layer::getInputCount() const
	{
		return inputCount;
	}

	size_t Layer::getOutputCount() const
	{
		return outputCount;
	}

	const std::vector<float>& Layer::getOutputs() const
	{
		return outputs;
	}

	const Matrix<float>& Layer::getWeights() const
	{
		return weights;
	}

	const std::vector<float>& Layer::getBiases() const
	{
		return biases;
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

			outputs[i] = (*activationFunc)(outputs[i]);
		}
	}

	float Layer::calculateSumLoss(const std::vector<float>& expectedOutputs)
	{
		float sumLoss = 0.0f;

		for (int i = 0; i < outputs.size(); i++)
		{
			sumLoss += (*lossFunc)(outputs[i], expectedOutputs[i]);
		}

		return sumLoss;
	}
}