#include <iostream>

#include "matrix.h"
#include "utils.h"
#include "layer.h"
#include "neuralNetwork.h"

int main()
{
	/*MachineLearning::Matrix<float> mat = {
		{1, 2, 3, 4}, 
		{5, 6, 7, 8}
	};

	for (int i = 0; i < mat.getRows(); i++)
	{
		for (int j = 0; j < mat.getColumns(); j++)
		{
			mat(i, j) = MachineLearning::Utils::getRandomFloat(0.0f, 20.0f);
		}
	}

	MachineLearning::Utils::printFloatMatrix(mat);*/

	MachineLearning::NeuralNetwork neuralNet = { 2, 4, 4, 3 };

	const int layerNum = 2;

	std::cout << "Inputs: " << neuralNet.getLayers()[layerNum].getInputCount() << std::endl;
	std::cout << "Outputs: " << neuralNet.getLayers()[layerNum].getOutputCount() << std::endl;

	std::cout << "\nWeights: \n";
	neuralNet.getLayers()[layerNum].getWeights().print();

	std::cout << "Biases: \n";
	for (auto v = neuralNet.getLayers()[layerNum].getBiases().begin(); v < neuralNet.getLayers()[layerNum].getBiases().end(); ++v)
		std::cout << *v << " ";
	std::cout << std::endl;

	return 0;
}