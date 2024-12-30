#include <iostream>

#include "utils.h"

int main()
{
	std::cout << "Started..." << std::endl;

	for (int i = 1; i <= 100; i++)
	{
		std::cout << "Random number " << i << ": " 
			<< MachineLearning::Utils::getRandomFloat(-0.2f, 0.2f) << "\n";
	}

	std::cout << "Done." << std::endl;

	return 0;
}