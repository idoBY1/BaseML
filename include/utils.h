#pragma once

#include "matrix.h"

namespace BaseML::Utils
{
	float getRandomFloat(float min, float max);

	float sigmoid(float input);
	float sigmoidDerivative(float neuronOutput);

	float squareError(float activation, float expected);
	float squareErrorDerivative(float activation, float expected);

	void printFloatMatrix(const Matrix<float>& mat);
}