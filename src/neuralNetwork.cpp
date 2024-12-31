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

	float NeuralNetwork::calculateSumLoss(const Matrix<float>& expectedOutputs) // TODO: fix compatibility with 2D matrices (after adding batches)
	{
		float sumLoss = 0.0f;

		Layer& lastLayer = layers[layers.size() - 1];

		for (int i = 0; i < lastLayer.getOutputCount(); i++)
		{
			sumLoss += (*lossFunc)((lastLayer.getOutputs())(i), expectedOutputs(i));
		}

		return sumLoss;
	}

	//float NeuralNetwork::calculateAverageLoss(const std::vector<Matrix<float>>& inputs, const std::vector<Matrix<float>>& expectedOutputs) // TODO: Modify to work with batches
	//{
	//	if (inputs.size() != expectedOutputs.size())
	//	{
	//		std::cout << "Inputs and expected outputs should have the same length!" << std::endl;
	//		throw std::runtime_error("Inputs size must match expected outputs size");
	//	}

	//	float sumLoss = 0.0f;

	//	for (int i = 0; i < expectedOutputs.size(); i++)
	//	{
	//		forwardPropagate(inputs[i]);
	//		sumLoss += layers[layers.size() - 1].calculateSumLoss(expectedOutputs[i]);
	//	}

	//	return sumLoss / expectedOutputs.size();
	//}

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

	void NeuralNetwork::learn(const Matrix<float>& inputs, const Matrix<float>& expectedOutputs, float learningRate) // TODO: expand to work with batches
	{
		forwardPropagate(inputs);
		backPropagation(inputs, expectedOutputs, learningRate);
	}
}