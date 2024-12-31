#include <iostream>

#include "matrix.h"
#include "utils.h"

int main()
{
	MachineLearning::Matrix<float> mat({
		1, 
		2,
		3, 
		4
		});

	MachineLearning::Utils::printFloatMatrix(mat);

	return 0;
}