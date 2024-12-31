#pragma once

#include <vector>

#include "layer.h"

namespace MachineLearning
{
	class NeuralNetwork
	{
	private:
		std::vector<Layer> layers;
		float (*lossFunc)(float, float), (*lossFuncDerivative)(float, float); // Function to minimize

	public:
		// Create a Neural Network with layers of the sizes specified in the initializer list
		NeuralNetwork(std::initializer_list<size_t> layerSizes);

		// Returns the layers of the Neural Network
		const std::vector<Layer>& getLayers() const;

		// Runs the input through the network
		const Matrix<float>& forwardPropagate(const Matrix<float>& inputs);

		// Calculates the sum of the loss function over all of the last layer's outputs
		float calculateSumLoss(const Matrix<float>& expectedOutputs);

		/*float calculateAverageLoss(const std::vector<Matrix<float>>& inputs, const std::vector<Matrix<float>>& expectedOutputs);*/
	};
}