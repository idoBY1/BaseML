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

	public:
		Layer(); // Default constructor for creating an empty object
		Layer(size_t numInputs, size_t numOutputs);
		Layer(size_t numInputs, size_t numOutputs, float (*activationFunction)(float), float (*activationFunctionDerivative)(float));

		size_t getInputCount();
		size_t getOutputCount();
		const std::vector<float>& getOutputs();

		void calculateOutputs(const std::vector<float>& inputs); // Forward propagation
	};
}