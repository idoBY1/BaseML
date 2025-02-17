#pragma once

#include <vector>

#include "Layer.h"
#include "Utils.h"

namespace BaseML
{
	class NeuralNetwork
	{
	private:
		std::vector<Layer> layers;
		Matrix networkInput;
		float (*lossFunc)(float, float), (*lossFuncDerivative)(float, float); // Function to minimize

		size_t learningTimestep;

	public:
		// Create an empty Neural Network
		NeuralNetwork();

		// Create a Neural Network with layers of the sizes specified in the initializer list
		NeuralNetwork(std::initializer_list<size_t> layerSizes);

		// Create a Neural Network with layers of the sizes specified in the 'layerSizes' initializer list.
		// Uses the activation functions provided for the hidden layers and last layer.
		NeuralNetwork(std::initializer_list<size_t> layerSizes, float (*hiddenActFunc)(float),
			float (*hiddenActFuncDerivative)(float), float (*outputActFunc)(float), float (*outputActFuncDerivative)(float));

		// Create a Neural Network with layers of the sizes specified in the initializer list. Use the 
		// activation function provided.
		NeuralNetwork(std::initializer_list<size_t> layerSizes, float (*activationFunction)(float),
			float (*activationFunctionDerivative)(float));

		// Create a Neural Network with layers of the sizes specified in the initializer list. Use the 
		// activation and loss functions provided.
		NeuralNetwork(std::initializer_list<size_t> layerSizes, float (*activationFunction)(float),
			float (*activationFunctionDerivative)(float), float (*lossFunction)(float, float), float (*lossFunctionDerivative)(float, float));

		// Returns the layers of the Neural Network
		const std::vector<Layer>& getLayers() const;

		// Returns the number of inputs of this network
		size_t getInputCount() const;

		// Returns the number of outputs of this network
		size_t getOutputCount() const;

		// Returns the output of the last layer
		const Matrix& getOutput() const;

		// Returns the index of neuron with highest output. Assumes that the network output contains one data point
		int getClassify();

		// Runs the input through the network
		const Matrix& forwardPropagate(const Matrix& inputs);

		// Calculates the sum of the loss function over all of the last layer's outputs
		float calculateSumLoss(const Matrix& expectedOutputs);

		// Calculate gradients and apply gradient descent on every layer of the Neural Network
		void backPropagation(const Matrix& inputs, const Matrix& expectedOutputs, float learningRate = 0.001f);

		// Pass the data through the Neural Network and perform gradient descent. Returns the loss
		float learn(const Matrix& inputs, const Matrix& expectedOutputs, float learningRate = 0.001f);

		// Pass the data through the Neural Network and perform gradient descent. Each pair of matrices represent 
		// a mini-batch where the first Matrix represents the inputs of the batch and the second Matrix represents 
		// the expected outputs of the batch. Each column represents a data point which means that the size of the 
		// matrices should be inputNeuronCount rows and batchSize columns for the first Matrix and outputNeuronCount 
		// rows and batchSize columns for the second Matrix (e.g. (inputs x batch) and (outputs x batch)).
		// Returns the loss of the last batch
		float learn(const std::vector<std::pair<Matrix, Matrix>>& data, float learningRate = 0.001f);

		// Save Neural Network to disk. Assumes a binary output stream
		void save(std::ofstream& outFile);

		// Save The Neuaral Network in its current state to the specified file (if the file doesn't exist, it will be created)
		void saveToFile(const char* fileName);

		// Save The Neuaral Network to a file. The name of the file will be generated automatically
		void saveParams(const char* networkName = "neural_network", float networkScore = -1.0f, bool includeTime = true);

		// Load Neural Network from disk. Assumes a binary input stream
		void load(std::ifstream& inFile, float (*hiddenActFunc)(float) = &Utils::leakyReLU,
			float (*hiddenActFuncDerivative)(float) = &Utils::leakyReLUDerivative, float (*activationFunction)(float) = &Utils::sigmoid,
			float (*activationFunctionDerivative)(float) = &Utils::sigmoidDerivative, float (*lossFunction)(float, float) = &Utils::squareError, 
			float (*lossFunctionDerivative)(float, float) = &Utils::squareErrorDerivative);

		// Try to load Neural Network from the specified file. Returns true if successful and false if something went wrong
		bool loadFromFile(const char* fileName, float (*hiddenActFunc)(float) = &Utils::leakyReLU,
			float (*hiddenActFuncDerivative)(float) = &Utils::leakyReLUDerivative, float (*activationFunction)(float) = &Utils::sigmoid,
			float (*activationFunctionDerivative)(float) = &Utils::sigmoidDerivative, float (*lossFunction)(float, float) = &Utils::squareError,
			float (*lossFunctionDerivative)(float, float) = &Utils::squareErrorDerivative);
	};
}