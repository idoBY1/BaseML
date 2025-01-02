#include "neuralNetwork.h"

#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <format>

#include "utils.h"

namespace BaseML
{
	NeuralNetwork::NeuralNetwork()
		:lossFunc(nullptr), lossFuncDerivative(nullptr)
	{
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes)
		:lossFunc(&Utils::squareError), lossFuncDerivative(&Utils::squareErrorDerivative)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end(); layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize);
		}
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes, float(*activationFunction)(float), float(*activationFunctionDerivative)(float))
		:lossFunc(&Utils::squareError), lossFuncDerivative(&Utils::squareErrorDerivative)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end(); layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize, activationFunction, activationFunctionDerivative);
		}
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes, float(*activationFunction)(float), float(*activationFunctionDerivative)(float), 
		float(*lossFunction)(float, float), float(*lossFunctionDerivative)(float, float))
		:lossFunc(lossFunction), lossFuncDerivative(lossFunctionDerivative)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end(); layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize, activationFunction, activationFunctionDerivative);
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

	void NeuralNetwork::save(std::ofstream& outFile)
	{
		int numOfLayers = layers.size();

		outFile.write(reinterpret_cast<const char*>(&numOfLayers), sizeof(numOfLayers));

		for (int i = 0; i < numOfLayers; i++)
		{
			layers[i].save(outFile);
		}
	}

	void NeuralNetwork::saveToFile(const char* fileName)
	{
		std::ofstream ofile;

		ofile.open(fileName, std::ios::binary | std::ios::out);

		save(ofile);

		ofile.close();
	}

	void NeuralNetwork::saveParams(const char* networkName, float networkScore, bool includeTime)
	{
		std::string fileName = networkName;

		if (networkScore != -1.0f)
			fileName += std::format("--Score_{}", networkScore);

		if (includeTime)
		{
			char tempTimeStr[50];
			time_t timestamp = time(NULL);
			struct tm* datetime = localtime(&timestamp);

			strftime(tempTimeStr, 50, "--%d-%m-%Y_%H-%M", datetime);

			fileName += tempTimeStr;
		}

		fileName += ".nn";

		saveToFile(fileName.c_str());
	}

	void NeuralNetwork::load(std::ifstream& inFile, float(*activationFunction)(float), float(*activationFunctionDerivative)(float), 
		float(*lossFunction)(float, float), float(*lossFunctionDerivative)(float, float))
	{
		lossFunc = lossFunction;
		lossFuncDerivative = lossFunctionDerivative;

		int numOfLayers;

		inFile.read(reinterpret_cast<char*>(&numOfLayers), sizeof(numOfLayers));

		layers = std::vector<Layer>(numOfLayers);

		for (int i = 0; i < numOfLayers; i++)
		{
			layers[i].load(inFile, activationFunction, activationFunctionDerivative);
		}
	}

	bool NeuralNetwork::loadFromFile(const char* fileName, float(*activationFunction)(float), float(*activationFunctionDerivative)(float),
		float(*lossFunction)(float, float), float(*lossFunctionDerivative)(float, float))
	{
		try {
			std::ifstream ifile;

			ifile.open(fileName, std::ios::binary | std::ios::in);

			load(ifile, activationFunction, activationFunctionDerivative, 
				lossFunction, lossFunctionDerivative);

			ifile.close();

			return true;
		}
		catch(...) {
			return false;
		}
	}
}