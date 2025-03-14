#include "RL/Player.h"

#include "Core/Matrix.h"

namespace BaseML::RL
{
	void Player::setAction(const Matrix& action)
	{
		currentAction = action;
	}

	void Player::setObservation(const Matrix& observation)
	{
		currentObservation = observation;
	}

	void Player::setReward(float reward)
	{
		currentReward = reward;
	}

	const Matrix& Player::getAction() const
	{
		return currentAction;
	}

	const Matrix& Player::getObservation() const
	{
		return currentObservation;
	}

	float Player::getReward() const
	{
		return currentReward;
	}
}