#include "layer.h"

#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>

#include "utils.h"

namespace BaseML
{
	Layer::Layer()
		:inputCount(0), outputCount(0), batchSize(1), weights(), biases(), outputs(), gradients(), activationFunc(nullptr), activationFuncDerivative(nullptr)
	{
	}

	Layer::Layer(size_t numInputs, size_t numOutputs)
		:inputCount(numInputs), outputCount(numOutputs), batchSize(1), weights(numOutputs, numInputs), biases(numOutputs, 1), outputs(numOutputs, batchSize), 
		gradients(numOutputs, batchSize), activationFunc(&Utils::sigmoid), activationFuncDerivative(&Utils::sigmoidDerivative)
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
		float(*activationFunctionDerivative)(float))
		:inputCount(numInputs), outputCount(numOutputs), batchSize(1), weights(numOutputs, numInputs), biases(numOutputs, 1), outputs(numOutputs, batchSize), 
		gradients(numOutputs, 1), activationFunc(activationFunction), activationFuncDerivative(activationFunctionDerivative)
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

	size_t Layer::getCurrentBatchSize() const
	{
		return batchSize;
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
#ifdef DEBUG
		if (inputCount != inputs.rowsCount())
		{
			std::cout << "Invalid input size for layer" << std::endl;
			throw std::runtime_error("Invalid input size for layer");
		}
#endif // DEBUG

		// Update batch size according to the input
		batchSize = inputs.columnsCount();

		// Reset outputs and resize to fit the input
		outputs = Matrix<float>(outputCount, batchSize);

		// Multiply and add matrices to calculate the activation of each neuron.
		// The result for each neuron is the sum of activations in the previous layer 
		// weighted by the weights of the connections to each neuron on the previous 
		// layer.
		outputs = (weights * inputs).addToColumns(biases);

		// Pass the results through the activation function
		for (int i = 0; i < outputs.size(); i++)
		{
			outputs(i) = (*activationFunc)(outputs(i));
		}
	}

	void Layer::calculateLastLayerGradients(const Matrix<float>& expectedOutputs, float(*lossFunctionDerivative)(float, float))
	{
#ifdef DEBUG
		if (outputCount != expectedOutputs.rowsCount() || outputs.columnsCount() != expectedOutputs.columnsCount())
		{
			std::cout << "Invalid expected output size for layer" << std::endl;
			throw std::runtime_error("Invalid expected output size for layer");
		}
#endif // DEBUG

		// Reset gradients and resize to fit the output
		gradients = Matrix<float>(outputCount, batchSize);

		// The Last layer bases its gradients on the loss function directly
		for (int i = 0; i < gradients.size(); i++)
		{
			gradients(i) = (*lossFunctionDerivative)(outputs(i), expectedOutputs(i)) * (*activationFuncDerivative)(outputs(i));
		}
	}

	void Layer::calculateGradients(const Layer& nextLayer)
	{
		// Reset gradients and resize to fit the output
		gradients = Matrix<float>(outputCount, batchSize);

		// Multiply nextLayer's weights (the weights connecting this layer of neurons and
		// the next later of neurons) with nextLayer's gradients. The result for each neuron 
		// will be the sum of gradients in the next layer weighted by their corresponding 
		// weights. This is the first part of the derivative.
		gradients = nextLayer.weights.transpose() * nextLayer.getGradients();

		// Add the derivative of the activation function to each neuron's gradient.
		// This is the second part of the derivative and the last shared part of the 
		// derivative shared by both the weights and the biases
		for (int i = 0; i < gradients.size(); i++)
		{
			gradients(i) = gradients(i) * (*activationFuncDerivative)(outputs(i));
		}
	}

	void Layer::gradientDescent(const Matrix<float>& previousLayerOutputs, float learningRate)
	{
		// Complete the gradient calculation, multiply by the learning-rate and subtract from
		// the currect weights. To get the final gradient for the weights, we multiply the shared 
		// gradients with the outputs of the neurons of the previous layer.
		weights = weights - ((gradients * previousLayerOutputs.transpose()) * learningRate);

		// Multiply by learning-rate and update biases. Sum the rows of the gradients to add 
		// all of the gradients from the batch to one update.
		biases = biases - (gradients.sumRows() * learningRate);
	}

	void Layer::save(std::ofstream& outFile)
	{
		outFile.write(reinterpret_cast<const char*>(&inputCount), sizeof(inputCount));
		outFile.write(reinterpret_cast<const char*>(&outputCount), sizeof(outputCount));

		weights.save(outFile);
		biases.save(outFile);
	}

	void Layer::load(std::ifstream& inFile, float(*activationFunction)(float), float(*activationFunctionDerivative)(float))
	{
		activationFunc = activationFunction;
		activationFuncDerivative = activationFunctionDerivative;

		inFile.read(reinterpret_cast<char*>(&inputCount), sizeof(inputCount));
		inFile.read(reinterpret_cast<char*>(&outputCount), sizeof(outputCount));

		weights.load(inFile);
		biases.load(inFile);
	}
}