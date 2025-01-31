#pragma once

#include "matrix.h"

namespace BaseML::Utils
{
	float getRandomFloat(float min, float max);

	float sigmoid(float input);
	float sigmoidDerivative(float neuronOutput);

	float leakyReLU(float input);
	float leakyReLUDerivative(float neuronOutput);

	float squareError(float activation, float expected);
	float squareErrorDerivative(float activation, float expected);
}