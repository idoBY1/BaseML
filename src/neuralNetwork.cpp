#include "neuralNetwork.h"

#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <format>

#include "UtilsFunctions.h"

namespace BaseML
{
	NeuralNetwork::NeuralNetwork()
		:lossFunc(nullptr), lossFuncDerivative(nullptr), networkInput(), learningTimestep(1)
	{
	}

	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes)
		:lossFunc(&Utils::squareError), lossFuncDerivative(&Utils::squareErrorDerivative), networkInput(), learningTimestep(1)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end() - 1; layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize);
		}

		layers.emplace_back(*(layerSizes.end() - 2), *(layerSizes.end() - 1), &Utils::sigmoid, &Utils::sigmoidDerivative);
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes)
		:lossFunc(&Utils::squareError), lossFuncDerivative(&Utils::squareErrorDerivative), networkInput(), learningTimestep(1)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end() - 1; layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize);
		}

		layers.emplace_back(*(layerSizes.end() - 2), *(layerSizes.end() - 1), &Utils::sigmoid, &Utils::sigmoidDerivative);
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes, float(*hiddenActFunc)(float), float(*hiddenActFuncDerivative)(float), float(*outputActFunc)(float), float(*outputActFuncDerivative)(float))
		:lossFunc(&Utils::squareError), lossFuncDerivative(&Utils::squareErrorDerivative), networkInput(), learningTimestep(1)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end() - 1; layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize, hiddenActFunc, hiddenActFuncDerivative);
		}

		layers.emplace_back(*(layerSizes.end() - 2), *(layerSizes.end() - 1), outputActFunc, outputActFuncDerivative);
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes, float(*activationFunction)(float), float(*activationFunctionDerivative)(float))
		:lossFunc(&Utils::squareError), lossFuncDerivative(&Utils::squareErrorDerivative), networkInput(), learningTimestep(1)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end(); layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize, activationFunction, activationFunctionDerivative);
		}
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> layerSizes, float(*activationFunction)(float), float(*activationFunctionDerivative)(float), 
		float(*lossFunction)(float, float), float(*lossFunctionDerivative)(float, float))
		:lossFunc(lossFunction), lossFuncDerivative(lossFunctionDerivative), networkInput(), learningTimestep(1)
	{
		layers.reserve(layerSizes.size() - 1);

		for (auto layerSize = layerSizes.begin() + 1; layerSize < layerSizes.end(); layerSize++)
		{
			layers.emplace_back(*(layerSize - 1), *layerSize, activationFunction, activationFunctionDerivative);
		}
	}

	void NeuralNetwork::setOutputActivationFunction(float(*activationFunction)(float), float(*activationFunctionDerivative)(float))
	{
		layers[layers.size() - 1].setActivationFunction(activationFunction, activationFunctionDerivative);
	}

	void NeuralNetwork::setHiddenActivationFunction(float(*activationFunction)(float), float(*activationFunctionDerivative)(float))
	{
		for (auto layer = layers.begin(); layer < layers.end() - 1; layer++)
		{
			layer->setActivationFunction(activationFunction, activationFunctionDerivative);
		}
	}

	const std::vector<Layer>& NeuralNetwork::getLayers() const
	{
		return layers;
	}

	size_t NeuralNetwork::getInputCount() const
	{
		return layers[0].getInputCount();
	}

	size_t NeuralNetwork::getOutputCount() const
	{
		return layers[layers.size() - 1].getOutputCount();
	}

	const Matrix& NeuralNetwork::getOutput() const
	{
		return layers[layers.size() - 1].getOutputs();
	}

	int NeuralNetwork::getClassify()
	{
		int maxIndex = 0;
		float maxOutput = 0.0f;

		auto outputs = getOutput();

		for (int i = 0; i < outputs.rowsCount(); i++)
		{
			if (outputs(i, 0) > maxOutput)
			{
				maxIndex = i;
				maxOutput = outputs(i, 0);
			}
		}

		return maxIndex;
	}

	const Matrix& NeuralNetwork::forwardPropagate(const Matrix& inputs)
	{
		networkInput = inputs;

		layers[0].calculateOutputs(&networkInput);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i].calculateOutputs(&(layers[i - 1].getOutputs()));
		}

		return layers[layers.size() - 1].getOutputs();
	}

	float NeuralNetwork::calculateSumLoss(const Matrix& expectedOutputs)
	{
		float sumLoss = 0.0f;

		Layer& lastLayer = layers[layers.size() - 1];

		for (int i = 0; i < lastLayer.getOutputs().size(); i++)
		{
			sumLoss += (*lossFunc)((lastLayer.getOutputs())(i), expectedOutputs(i));
		}

		return (sumLoss / lastLayer.getCurrentBatchSize()); // if multiple data points in the batch, return the average loss
	}

	void NeuralNetwork::backPropagationToTarget(const Matrix& expectedOutputs, float learningRate)
	{
		// Calculate gradients
		layers[layers.size() - 1].calculateLastLayerGradientsToTarget(expectedOutputs, lossFuncDerivative);

		for (int i = layers.size() - 2; i >= 0; i--)
		{
			layers[i].calculateGradients(layers[i + 1]);
		}

		// Update parameters
		layers[0].adamGradientDescent(learningRate, learningTimestep);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i].adamGradientDescent(learningRate, learningTimestep);
		}

		learningTimestep++;
	}

	void NeuralNetwork::backPropagation(const Matrix& externalGradients, float learningRate)
	{
		// Calculate gradients
		layers[layers.size() - 1].calculateLastLayerGradients(externalGradients);

		for (int i = layers.size() - 2; i >= 0; i--)
		{
			layers[i].calculateGradients(layers[i + 1]);
		}

		// Update parameters
		layers[0].adamGradientDescent(learningRate, learningTimestep);

		for (int i = 1; i < layers.size(); i++)
		{
			layers[i].adamGradientDescent(learningRate, learningTimestep);
		}

		learningTimestep++;
	}

	float NeuralNetwork::learn(const Matrix& inputs, const Matrix& expectedOutputs, float learningRate)
	{
		forwardPropagate(inputs);
		backPropagationToTarget(expectedOutputs, learningRate);

		return calculateSumLoss(expectedOutputs);
	}

	float NeuralNetwork::learn(const std::vector<std::pair<Matrix, Matrix>>& data, float learningRate)
	{
		float currLoss;

		for (int i = 0; i < data.size(); i++)
		{
			currLoss = learn(data[i].first, data[i].second, learningRate);
		}

		return currLoss;
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

	void NeuralNetwork::load(std::ifstream& inFile, float (*hiddenActFunc)(float), float (*hiddenActFuncDerivative)(float),
		float(*activationFunction)(float), float(*activationFunctionDerivative)(float),
		float(*lossFunction)(float, float), float(*lossFunctionDerivative)(float, float))
	{
		lossFunc = lossFunction;
		lossFuncDerivative = lossFunctionDerivative;

		int numOfLayers;

		inFile.read(reinterpret_cast<char*>(&numOfLayers), sizeof(numOfLayers));

		layers = std::vector<Layer>(numOfLayers);

		for (int i = 0; i < numOfLayers - 1; i++)
		{
			layers[i].load(inFile, hiddenActFunc, hiddenActFuncDerivative);
		}

		layers[numOfLayers - 1].load(inFile, activationFunction, activationFunctionDerivative);
	}

	bool NeuralNetwork::loadFromFile(const char* fileName, float (*hiddenActFunc)(float), float (*hiddenActFuncDerivative)(float),
		float(*activationFunction)(float), float(*activationFunctionDerivative)(float),
		float(*lossFunction)(float, float), float(*lossFunctionDerivative)(float, float))
	{
		try {
			std::ifstream ifile;

			ifile.open(fileName, std::ios::binary | std::ios::in);

			load(ifile, hiddenActFunc, hiddenActFuncDerivative, 
				activationFunction, activationFunctionDerivative,
				lossFunction, lossFunctionDerivative);

			ifile.close();

			return true;
		}
		catch(...) {
			return false;
		}
	}
}