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
		std::vector<float> biases, outputs, gradients;
		float (*activationFunc)(float), (*activationFuncDerivative)(float);
		float (*lossFunc)(float, float); // Function to minimize

	public:
		Layer(); // Default constructor for creating an empty object
		Layer(size_t numInputs, size_t numOutputs); // Defaults to the sigmoid activation function
		Layer(size_t numInputs, size_t numOutputs, float (*activationFunction)(float), 
			float (*activationFunctionDerivative)(float), float (*lossFunction)(float, float));

		size_t getInputCount();
		size_t getOutputCount();
		const std::vector<float>& getOutputs();

		void calculateOutputs(const std::vector<float>& inputs); // Forward propagation
		float calculateSumLoss(const std::vector<float>& expectedOutputs);
	};
}