#include "neuralNetwork.h"

#include <vector>
#include <cstdlib>
#include <ctime>

#include "utils.h"

namespace MachineLearning
{
	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes)
		:lossFunc(&Utils::squareError), lossFuncDerivative(&Utils::squareErrorDerivative)
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

	const Matrix<float>& NeuralNetwork::getOutput() const
	{
		return layers[layers.size() - 1].getOutputs();
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

	float NeuralNetwork::calculateSumLoss(const Matrix<float>& expectedOutputs)
	{
		float sumLoss = 0.0f;

		Layer& lastLayer = layers[layers.size() - 1];

		for (int i = 0; i < lastLayer.getOutputs().size(); i++)
		{
			sumLoss += (*lossFunc)((lastLayer.getOutputs())(i), expectedOutputs(i));
		}

		return (sumLoss / lastLayer.getCurrentBatchSize()); // if multiple data points in the batch, return the average loss
	}

	void NeuralNetwork::backPropagation(const Matrix<float>& inputs, const Matrix<float>& expectedOutputs, float learningRate)
	{
		// Calculate gradients
		layers[layers.size() - 1].calculateLastLayerGradients(expectedOutputs, lossFuncDerivative);

		for (int i = layers.size() - 2; i >= 0; i--)
		{
			layers[i].calculateGradients(layers[i + 1]);
		}

		// Update parameters
		layers[0].gradientDescent(inputs, learningRate);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i].gradientDescent(layers[i - 1].getOutputs(), learningRate);
		}
	}

	void NeuralNetwork::learn(const Matrix<float>& inputs, const Matrix<float>& expectedOutputs, float learningRate)
	{
		forwardPropagate(inputs);
		backPropagation(inputs, expectedOutputs, learningRate);
	}

	void NeuralNetwork::learn(const std::vector<std::pair<Matrix<float>, Matrix<float>>>& data, float learningRate)
	{
		for (int i = 0; i < data.size(); i++)
		{
			learn(data[i].first, data[i].second, learningRate);
		}
	}
}