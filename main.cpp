#include <iostream>

#include "matrix.h"
#include "layer.h"

int main()
{
	MachineLearning::Matrix<float> mat(3, 2);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			mat(i, j) = i * 2 + j;
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			std::cout << mat(i, j) << "\n";
		}
	}

	return 0;
}