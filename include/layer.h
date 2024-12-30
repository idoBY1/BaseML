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
		Matrix<float> weights;
		std::vector<float> biases;
		std::vector<float> outputs;
		std::vector<float> gradients;

	public:
		Layer(); // Default constructor for creating an empty object
		Layer(size_t numInputs, size_t numOutputs);

		size_t getInputCount();
		size_t getOutputCount();
		const std::vector<float>& getOutputs();

		float sigmoid(float input); // Activation function
		float sigmoidDerivative(float neuronOutput); // The derivative of the activation function

		void calculateOutputs(const std::vector<float>& inputs); // Forward propagation

		
	};
}