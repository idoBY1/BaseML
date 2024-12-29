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
		NeuralNetwork(std::vector<size_t> layerSizes);
		const std::vector<float>& forwardPropagate(const std::vector<float>& inputs);
	};
}