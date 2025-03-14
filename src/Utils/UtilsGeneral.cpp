#include "Utils/UtilsGeneral.h"

#include <cmath>

#include "Core/Matrix.h"

namespace BaseML::Utils
{
	Matrix zScoreNormalize(const Matrix& mat)
	{
		if (mat.size() == 0)
			return mat;

		float mean = 0.0f, variance = 0.0f, stddev;

		for (int i = 0; i < mat.size(); i++)
		{
			mean += mat(i);
			variance += mat(i) * mat(i);
		}

		mean /= mat.size();

		variance /= mat.size();
		variance -= mean * mean;

		stddev = std::sqrt(variance);

		Matrix norm(mat);

		#pragma omp parallel for
		for (int i = 0; i < norm.size(); i++)
		{
			norm(i) = (norm(i) - mean) / (stddev + 1.0e-8f);
		}

		return norm;
	}
}