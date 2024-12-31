#pragma once

#include <vector>

#include "matrix.h"

#define MAX_INIT_VAL 0.2f
#define MIN_INIT_VAL -0.2f

namespace MachineLearning
{
	class Layer
	{
	private:
		size_t inputCount, outputCount;
		Matrix<float> weights, biases, outputs, gradients;
		float (*activationFunc)(float), (*activationFuncDerivative)(float);
		float (*lossFunc)(float, float); // Function to minimize

	public:
		// Default constructor for creating an empty object
		Layer(); 

		// Creates a layer with 'numInputs' inputs and 'numOutputs' outputs.
		// Defaults to the sigmoid activation function with squareCost as the lost function
		Layer(size_t numInputs, size_t numOutputs);

		// Creates a layer with 'numInputs' inputs and 'numOutputs' outputs. 
		Layer(size_t numInputs, size_t numOutputs, float (*activationFunction)(float), 
			float (*activationFunctionDerivative)(float), float (*lossFunction)(float, float));

		// Returns the number of inputs of this layer
		size_t getInputCount() const;

		// Returns the number of outputs of this layer
		size_t getOutputCount() const;

		// Returns the outputs of this layer
		const Matrix<float>& getOutputs() const;

		// Returns the weights of this layer
		const Matrix<float>& getWeights() const;

		// Returns the biases of this layer
		const Matrix<float>& getBiases() const;

		// Perform forward propagation on this layer with the specified inputs
		void calculateOutputs(const Matrix<float>& inputs); 

		// Calculates the sum of the loss function over all of the layer's outputs
		float calculateSumLoss(const Matrix<float>& expectedOutputs);
	};
}