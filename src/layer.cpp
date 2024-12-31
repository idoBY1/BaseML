#include "layer.h"

#include <vector>
#include <cstdlib>
#include <cmath>

#include "utils.h"

namespace MachineLearning
{
	Layer::Layer()
		:inputCount(0), outputCount(0), weights(), biases(), outputs(), gradients(), activationFunc(nullptr), activationFuncDerivative(nullptr)
	{
	}

	Layer::Layer(size_t numInputs, size_t numOutputs)
		:inputCount(numInputs), outputCount(numOutputs), weights(numOutputs, numInputs), biases(numOutputs, 1), outputs(numOutputs, 1), 
		gradients(1, numOutputs), activationFunc(&Utils::sigmoid), activationFuncDerivative(&Utils::sigmoidDerivative)
	{
		// Initialize the biases and weights with random values

		for (int i = 0; i < numOutputs; i++)
		{
			// Generate a number between MIN_INIT_VAL and MAX_INIT_VAL (the number is a float)
			biases(i) = Utils::getRandomFloat(MIN_INIT_VAL, MAX_INIT_VAL);

			for (int j = 0; j < numInputs; j++)
			{
				weights(i, j) = Utils::getRandomFloat(MIN_INIT_VAL, MAX_INIT_VAL);
			}
		}
	}

	Layer::Layer(size_t numInputs, size_t numOutputs, float(*activationFunction)(float), 
		float(*activationFunctionDerivative)(float), float (*lossFunction)(float, float))
		:inputCount(numInputs), outputCount(numOutputs), weights(numOutputs, numInputs), biases(numOutputs, 1), outputs(numOutputs, 1), gradients(numOutputs, 1), 
		activationFunc(activationFunction), activationFuncDerivative(activationFunctionDerivative)
	{
		// Initialize the biases and weights with random values

		for (int i = 0; i < numOutputs; i++)
		{
			// Generate a number between MIN_INIT_VAL and MAX_INIT_VAL (the number is a float)
			biases(i) = Utils::getRandomFloat(MIN_INIT_VAL, MAX_INIT_VAL);

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

	const Matrix<float>& Layer::getOutputs() const
	{
		return outputs;
	}

	const Matrix<float>& Layer::getWeights() const
	{
		return weights;
	}

	const Matrix<float>& Layer::getBiases() const
	{
		return biases;
	}

	const Matrix<float>& Layer::getGradients() const
	{
		return gradients;
	}

	void Layer::calculateOutputs(const Matrix<float>& inputs)
	{
		// Multiply and add matrices to calculate the activation of each neuron.
		// The result for each neuron is the sum of activations in the previous layer 
		// weighted by the weights of the connections to each neuron on the previous 
		// layer.
		outputs = (weights * inputs) + biases;

		// Pass the results through the activation function
		for (int i = 0; i < outputCount; i++)
		{
			outputs(i) = (*activationFunc)(outputs(i));
		}
	}

	void Layer::calculateLastLayerGradients(const Matrix<float>& expectedOutputs, float(*lossFunctionDerivative)(float, float))
	{
		// The Last layer bases its gradients on the loss function directly
		for (int i = 0; i < outputCount; i++)
		{
			gradients(i) = (*lossFunctionDerivative)(outputs(i), expectedOutputs(i)) * (*activationFuncDerivative)(outputs(i));
		}
	}

	void Layer::calculateGradients(const Layer& nextLayer)
	{
		// Multiply nextLayer's gradients matrix with nextLayers weights matrix to get this layer's gradients
		// (Notice how in this operation the weights are on the right instead of on the left because we are 
		// going backwards in the layers. The gradients matrix is also rotated from the repressentation of the 
		// inputs and outputs in forward propagation to compensate for the direction of the data propagation)
		// The result for each neuron is the sum of all gradients in the next layer weighted by the weights 
		// of the connections to each neuron on the next layer.
		gradients = nextLayer.getGradients() * nextLayer.getWeights();

		// Add the derivative of the activation function to each neuron's gradient
		for (int i = 0; i < outputCount; i++)
		{
			gradients(i) = gradients(i) * (*activationFuncDerivative)(outputs(i));
		}
	}

	void Layer::gradientDescent(float learningRate)
	{
		// TODO: implement
	}
}