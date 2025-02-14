#pragma once

#include <vector>

#include "Matrix.h"
#include "Utils.h"

namespace BaseML
{
	class Layer
	{
	private:
		size_t inputCount, outputCount, batchSize;
		Matrix weights, biases, outputs, gradients;
		const Matrix* inputRef; // A pointer to the output of last layer. This class doesn't manage this memory!
		float (*activationFunc)(float), (*activationFuncDerivative)(float);

		// Adam Optimizer matrices
		Matrix mWeights, vWeights, mBiases, vBiases;

	public:
		// Default constructor for creating an empty object
		Layer(); 

		// Creates a layer with 'numInputs' inputs and 'numOutputs' outputs.
		// Defaults to the sigmoid activation function with squareCost as the lost function
		Layer(size_t numInputs, size_t numOutputs);

		// Creates a layer with 'numInputs' inputs and 'numOutputs' outputs. 
		Layer(size_t numInputs, size_t numOutputs, float (*activationFunction)(float), 
			float (*activationFunctionDerivative)(float));

		// Returns the number of inputs of this layer
		size_t getInputCount() const;

		// Returns the number of outputs of this layer
		size_t getOutputCount() const;

		// Returns the number data points the layer is currently configured to process together
		size_t getCurrentBatchSize() const;

		// Returns the outputs of this layer
		const Matrix& getOutputs() const;

		// Returns the weights of this layer
		const Matrix& getWeights() const;

		// Returns the biases of this layer
		const Matrix& getBiases() const;

		// Returns the gradients of this layer
		const Matrix& getGradients() const;

		// Perform forward propagation on this layer with the specified inputs
		void calculateOutputs(const Matrix* inputs); 

		// Caculate the gradients of the last layer based on the loss function and the expected outputs
		void calculateLastLayerGradients(const Matrix& expectedOutputs, float (*lossFunctionDerivative)(float, float));

		// Caculate the gradients of this layer based on the gradients of the next layer
		void calculateGradients(const Layer& nextLayer);

		// Update the parameters according to the gradients to minimize the loss function
		void gradientDescent(float learningRate);

		// Update the parameters according to the gradients to minimize the loss function using Adam Optimizer.
		// 'timestep' is the number of times the layer's parameters have been updated before. 'beta1' controlls 
		// the decay rate of the first moment (m) and 'beta2' controlls the decay rate of the second moment (v).
		// 'epsilon' is a small positive constant to avoid devision by zero. 
		void adamGradientDescent(float learningRate, size_t timestep, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1.0e-7f);

		// Save Layer to disk. Assumes a binary output stream
		void save(std::ofstream& outFile);

		// Load Layer from disk. Assumes a binary input stream
		void load(std::ifstream& inFile, float (*activationFunction)(float) = &Utils::sigmoid,
			float (*activationFunctionDerivative)(float) = &Utils::sigmoidDerivative);
	};
}