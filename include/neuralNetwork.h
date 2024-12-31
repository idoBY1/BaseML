#pragma once

#include <vector>

#include "layer.h"

namespace MachineLearning
{
	class NeuralNetwork
	{
	private:
		std::vector<Layer> layers;

	public:
		// Create a Neural Network with layers of the sizes specified in the vector
		NeuralNetwork(std::vector<size_t> layerSizes);

		// Create a Neural Network with layers of the sizes specified in the initializer list
		NeuralNetwork(std::initializer_list<size_t> layerSizes);

		// Returns the layers of the Neural Network
		const std::vector<Layer>& getLayers() const;

		float calculateAverageLoss(const Matrix<float>& expectedOutputs);

		const std::vector<float>& forwardPropagate(const std::vector<float>& inputs);
	};
}