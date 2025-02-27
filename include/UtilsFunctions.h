#pragma once

namespace BaseML::Utils
{
	float sigmoid(float input);
	float sigmoidDerivative(float neuronOutput);

	float leakyReLU(float input);
	float leakyReLUDerivative(float neuronOutput);

	float squareError(float activation, float expected);
	float squareErrorDerivative(float activation, float expected);
}