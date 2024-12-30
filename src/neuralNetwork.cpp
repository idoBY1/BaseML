#include "neuralNetwork.h"

#include <vector>
#include <cstdlib>
#include <ctime>

namespace MachineLearning
{
	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes)
		:layers(layerSizes.size() - 1)
	{
		// The first layer is just inputs so there is no need to save it inside layers
		for (int i = 1; i < layerSizes.size(); i++)
		{
			layers.emplace_back(layerSizes[i - 1], layerSizes[i]);
		}
	}
	
	float NeuralNetwork::calculateAverageLoss(const Matrix<float>& expectedOutputs)
	{
		return 0.0f; // TODO: implement function!!
	}

	const std::vector<float>& NeuralNetwork::forwardPropagate(const std::vector<float>& inputs)
	{
		layers[0].calculateOutputs(inputs);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i].calculateOutputs(layers[i - 1].getOutputs());
		}

		return layers[layers.size() - 1].getOutputs();
	}
}