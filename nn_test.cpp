#define DEBUG

#include <iostream>
#include <fstream>

#include "matrix.h"
#include "utils.h"
#include "layer.h"
#include "neuralNetwork.h"

void trainXOR()
{
	MachineLearning::NeuralNetwork neuralNet = { 2, 3, 1 };

	MachineLearning::Matrix<float> inputs = MachineLearning::Matrix<float>({ {0, 0}, {0, 1}, {1, 0}, {1, 1} }).transpose();
	MachineLearning::Matrix<float> expectedOutputs({ 0, 1, 1, 0 }, false);

	neuralNet.forwardPropagate({ 0, 0 });
	std::cout << "00 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 0, 1 });
	std::cout << "01 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 0 });
	std::cout << "10 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 1 });
	std::cout << "11 -> ";
	neuralNet.getOutput().print();

	std::cout << "started learning..." << std::endl;

	for (int i = 0; i < 1000; i++)
	{
		neuralNet.learn(inputs, expectedOutputs, 1);
	}

	std::cout << "finished learning.\n" << std::endl;

	neuralNet.forwardPropagate({ 0, 0 });
	std::cout << "00 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 0, 1 });
	std::cout << "01 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 0 });
	std::cout << "10 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 1 });
	std::cout << "11 -> ";
	neuralNet.getOutput().print();

	// Save to disk
	neuralNet.saveParams();
}

void testXOR()
{
	MachineLearning::NeuralNetwork neuralNet;

	char fileName[50] = "neural_network--02-01-2025_17-36.nn";

	if (!neuralNet.loadFromFile(fileName))
	{
		std::cout << "Failed to open file '" << fileName << "'" << std::endl;
		return;
	}

	neuralNet.forwardPropagate({ 0, 0 });
	std::cout << "00 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 0, 1 });
	std::cout << "01 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 0 });
	std::cout << "10 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 1 });
	std::cout << "11 -> ";
	neuralNet.getOutput().print();
}

void testTrainingXOR()
{
	MachineLearning::NeuralNetwork neuralNet = { 2, 3, 1 };

	MachineLearning::Matrix<float> inputs = MachineLearning::Matrix<float>({ {0, 0}, {0, 1}, {1, 0}, {1, 1} }).transpose();
	MachineLearning::Matrix<float> expectedOutputs({ 0, 1, 1, 0 }, false);

	neuralNet.forwardPropagate({ 0, 0 });
	std::cout << "00 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 0, 1 });
	std::cout << "01 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 0 });
	std::cout << "10 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 1 });
	std::cout << "11 -> ";
	neuralNet.getOutput().print();

	std::cout << "started learning..." << std::endl;

	for (int i = 0; i < 1000; i++)
	{
		neuralNet.learn(inputs, expectedOutputs, 1);
	}

	std::cout << "finished learning.\n" << std::endl;

	neuralNet.forwardPropagate({ 0, 0 });
	std::cout << "00 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 0, 1 });
	std::cout << "01 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 0 });
	std::cout << "10 -> ";
	neuralNet.getOutput().print();

	neuralNet.forwardPropagate({ 1, 1 });
	std::cout << "11 -> ";
	neuralNet.getOutput().print();
}

int main()
{
	testTrainingXOR();

	return 0;
}