#include "neuralNetwork.h"

#include <vector>
#include <cstdlib>
#include <ctime>

namespace MachineLearning
{
	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes)
	{
		layers.reserve(layerSizes.size() - 1);

		// The first layer is just inputs so there is no need to save it inside layers
		for (int i = 1; i < layerSizes.size(); i++)
		{
			layers.emplace_back(layerSizes[i - 1], layerSizes[i]);
		}
	}
	
	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end(); layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize);
		}
	}

	const std::vector<Layer>& NeuralNetwork::getLayers() const
	{
		return layers;
	}

	float NeuralNetwork::calculateAverageLoss(const Matrix<float>& expectedOutputs)
	{
		return 0.0f; // TODO: implement function!!
	}

	const Matrix<float>& NeuralNetwork::forwardPropagate(const Matrix<float>& inputs)
	{
		layers[0].calculateOutputs(inputs);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i].calculateOutputs(layers[i - 1].getOutputs());
		}

		return layers[layers.size() - 1].getOutputs();
	}
}