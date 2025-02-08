#define DEBUG

#include <iostream>
#include <fstream>
#include <chrono>

#include "matrix.h"
#include "utils.h"
#include "layer.h"
#include "neuralNetwork.h"

void trainXOR()
{
	BaseML::NeuralNetwork neuralNet = { 2, 3, 1 };

	BaseML::Matrix inputs = BaseML::Matrix({ {0, 0}, {0, 1}, {1, 0}, {1, 1} }).transpose();
	BaseML::Matrix expectedOutputs({ 0, 1, 1, 0 }, false);

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
	BaseML::NeuralNetwork neuralNet;

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
	BaseML::NeuralNetwork neuralNet = { 2, 50, 50, 1 };

	BaseML::Matrix inputs = BaseML::Matrix({ {0, 0}, {0, 1}, {1, 0}, {1, 1} }, true);
	BaseML::Matrix expectedOutputs({ 0, 1, 1, 0 }, false);

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

	float loss;

	auto last = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10000; i++)
	{
		loss = neuralNet.learn(inputs, expectedOutputs, 1);

		//if (i % 100 == 0) // every 100 iterations
		//{
		//	std::cout << "Current loss is: " << loss << std::endl;
		//}
	}
	auto finalTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last);

	std::cout << "Finished learning." << std::endl;
	std::cout << "Final loss is: " << loss << std::endl;
	std::cout << "Final time: " << finalTime << " milliseconds\n" << std::endl;

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