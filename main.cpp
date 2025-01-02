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
}

int main()
{
	/*std::ofstream ofile;

	ofile.open("matrixTest.dat", std::ios::binary | std::ios::out);

	MachineLearning::Matrix<float> mat = {
		{1, 2, 3}, 
		{4, 5, 3.5},
		{9, 8, 7},
		{10, 12, 11}
	};

	mat.save(ofile);

	mat.print();

	ofile.close();*/

	std::ifstream ifile;

	ifile.open("matrixTest.dat", std::ios::binary | std::ios::in);

	MachineLearning::Matrix<float> mat;

	mat.load(ifile);

	mat.print();

	ifile.close();

	return 0;
}