#pragma once

#include <vector>

#include "matrix.h"

#define MAX_INIT_VAL 0.1

namespace MachineLearning
{
	class Layer
	{
	private:
		size_t inputCount, outputCount;
		Matrix<float> weights;
		std::vector<float> biases;
		std::vector<float> outputs;

	public:
		Layer(); // Default constructor for creating an empty object
		Layer(size_t numInputs, size_t numOutputs);
		void calculateOutputs(const std::vector<float>& inputs);

		size_t getInputCount();
		size_t getOutputCount();
		const std::vector<float>& getOutputs();
	};
}