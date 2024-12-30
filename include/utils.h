#pragma once

namespace MachineLearning::Utils
{
	float getRandomFloat(float min, float max);

	float sigmoid(float input);
	float sigmoidDerivative(float neuronOutput);

	float squareError(float activation, float expected);
}