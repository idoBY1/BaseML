#pragma once

#include "Matrix.h"

namespace BaseML
{
	class IEnvironment
	{
		// Update the environment's state. 'deltaTime' is the amount of time to assume that 
		// have passed since the last update.
		virtual void update(float deltaTime);
	};
}