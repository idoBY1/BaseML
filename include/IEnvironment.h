#pragma once

#include "Matrix.h"

namespace BaseML
{
	class IEnvironment
	{
	public:
		virtual ~IEnvironment() = default; // Default destructor.

		virtual IEnvironment* clone() const = 0; // clone method for polymorphic copying (switch to unique_ptr)

		// Update the environment's state. 'deltaTime' is the amount of time to assume that 
		// have passed since the last update.
		virtual void update(float deltaTime) = 0;

		// Get the current state of the player as a column matrix.
		virtual const Matrix& getState(const char* playerId) const = 0;

		// Set the action that the player will perform in the next update.
		virtual void setAction(const char* playerId, const Matrix& action) = 0;

		// Get the reward of the current state of the player.
		virtual float getReward(const char* playerId) const = 0;

		// Initialize the environment.
		virtual void initialize() = 0;

		// Closes the environment's resources.
		virtual void close() = 0;

		// Resets the environment back to the initial state (this function expects the environment 
		// to be already initialized).
		virtual void reset() = 0;

		// Display the current state of the environment.
		virtual void render() = 0;
	};
}