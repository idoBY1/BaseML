#define DEBUG

#include <iostream>

#include "matrix.h"
#include "utils.h"
#include "layer.h"
#include "neuralNetwork.h"

int main()
{
	/*MachineLearning::Matrix<float> mat1 = {
		{1, 2, 3, 4}, 
		{5, 6, 7, 8},
		{1, 8, 3, 2}
	};

	MachineLearning::Matrix<float> mat2 = {
		{5, 3},
		{5, 2}, 
		{7, 5},
		{2, 9}
	};

	(mat1 + mat2).print();*/

	MachineLearning::NeuralNetwork neuralNet = { 2, 3, 1 };

	std::vector<MachineLearning::Matrix<float>> inputsVec = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	std::vector<MachineLearning::Matrix<float>> expectedOutputsVec = { {0}, {1}, {1}, {0} };

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

	for (int i = 0; i < 500; i++)
	{
		for (int j = 0; j < inputsVec.size(); j++)
		{
			neuralNet.learn(inputsVec[j], expectedOutputsVec[j], 10);
		}
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

	return 0;
}