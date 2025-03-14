#pragma once

#include "Core/Matrix.h"

namespace BaseML::RL
{
	class Player
	{
	private:
		Matrix currentAction, currentObservation;
		float currentReward;

	public:
		void setAction(const Matrix& action);
		void setObservation(const Matrix& observation);
		void setReward(float reward);

		const Matrix& getAction() const;
		const Matrix& getObservation() const;
		float getReward() const;
	};
}