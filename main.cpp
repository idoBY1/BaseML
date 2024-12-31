#define DEBUG

#include <iostream>

#include "matrix.h"
#include "utils.h"
#include "layer.h"
#include "neuralNetwork.h"

int main()
{
	MachineLearning::Matrix<float> mat1 = {
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

	(mat1 + mat2).print();

	/*MachineLearning::NeuralNetwork neuralNet = { 2, 4, 4, 3 };

	const int layerNum = 2;

	std::cout << "Inputs: " << neuralNet.getLayers()[layerNum].getInputCount() << std::endl;
	std::cout << "Outputs: " << neuralNet.getLayers()[layerNum].getOutputCount() << std::endl;

	std::cout << "\nWeights: \n";
	neuralNet.getLayers()[layerNum].getWeights().print();

	std::cout << "Biases: \n";
	neuralNet.getLayers()[layerNum].getBiases().print();
	std::cout << std::endl;*/

	return 0;
}