#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <Windows.h>

#include "UtilsRandom.h"
#include "Environment.h"
#include "Matrix.h"

namespace BaseML::RL::Tests {

    class CartPoleEnvironment : public Environment {
    private:
        // List of players (only one agent in this environment)
        std::vector<std::string> players = { "agent" };

        // Environment state: cart position (x), cart velocity (v),
        // pole angle (theta), and pole angular velocity (omega)
        float x, v, theta, omega;
        // Last applied force (action)
        float force;
        // Count of simulation steps
        int steps;
        // Maximum allowed steps per episode
        const int maxSteps = 500;
        // Termination flag for the episode
        bool done;
        // Initialization flag
        bool initialized;

        // Constants for dynamics
        const float gravity = 9.8f;
        const float masscart = 1.0f;
        const float masspole = 0.1f;
        const float totalMass = masscart + masspole;
        // Half the pole's length (for simplicity)
        const float poleLength = 0.5f;
        const float poleMassLength = masspole * poleLength;

        // The state is represented as a column vector: [x, v, theta, omega]^T
        Matrix stateMatrix;

        // Helper to update the stateMatrix with current state variables
        void updateStateMatrix() {
            // Re-create the column vector using a single-dimensional initializer list.
            // The second parameter 'true' indicates a column vector.
            stateMatrix = Matrix({ x, v, theta, omega }, true);
        }

    public:
        // Constructor: observation dimension is 4 and action dimension is 1.
        CartPoleEnvironment()
            : Environment(4, 1),
            x(0.0f), v(0.0f), theta(0.0f), omega(0.0f),
            force(0.0f), steps(0), done(false), initialized(false),
            stateMatrix({ 0.0f, 0.0f, 0.0f, 0.0f }, true)
        {
        }

        // Return the list of players (only one in this case)
        const std::vector<std::string>& getPlayers() const override {
            return players;
        }

        // Initialize the environment (reset state)
        void initialize() override {
            reset();
            initialized = true;
        }

        bool isInitialized() override {
            return initialized;
        }

        // Close any resources (nothing to do in this simple example)
        void close() override {
            initialized = false;
        }

        // Reset the environment to the initial state
        void reset() override {
            x = Utils::getRandomFloat(-0.05f, 0.05f);
            v = Utils::getRandomFloat(-0.05f, 0.05f);
            theta = Utils::getRandomFloat(-0.05f, 0.05f);
            omega = Utils::getRandomFloat(-0.05f, 0.05f);
            force = 0.0f;
            steps = 0;
            done = false;
            updateStateMatrix();
        }

        // Return true if the episode is finished (terminated)
        bool isFinished() override {
            return done;
        }

        // Render the current state (here we simply print it to the console)
        void render() override {
            /*std::cout << "Step: " << steps
                << ", x: " << x
                << ", v: " << v
                << ", theta: " << theta
                << ", omega: " << omega << std::endl;*/
            
            int temp = (int)((theta + 0.209f) / (0.418f / 11.0f));

            for (int i = 0; i < 11; i++)
            {
                if (temp == i)
                    std::cout << "*";
                else
                    std::cout << "_";
            }

            std::cout << std::endl;
        }

        // Return the current state as a column Matrix
        const Matrix& getState(const char* playerId) const override {
            return stateMatrix;
        }

        // Set the action (force) for the agent. Expects a 1x1 Matrix.
        void setAction(const char* playerId, const Matrix& action) override {
            // Here we assume action(0) extracts the first (and only) element.
            force = action(0);
        }

        // Return the reward for the current state.
        // In this simple version, we give a reward of +1 for each timestep the pole remains upright.
        float getReward(const char* playerId) const override {
            return done ? 0.0f : 1.0f;
        }

        // Update the environment state by one timestep (deltaTime seconds)
        void update(float deltaTime) override {
            if (done)
                return;

            // Compute cosine and sine of the pole angle.
            float costheta = std::cos(theta);
            float sintheta = std::sin(theta);

            // For the dynamics, calculate a temporary variable using the applied force.
            float temp = (force + poleMassLength * omega * omega * sintheta) / totalMass;
            // Compute the angular acceleration (theta acceleration)
            float thetaAcc = (gravity * sintheta - costheta * temp)
                / (poleLength * (4.0f / 3.0f - masspole * costheta * costheta / totalMass));
            // Compute the linear acceleration (x acceleration)
            float xAcc = temp - poleMassLength * thetaAcc * costheta / totalMass;

            // Euler integration to update state variables.
            x += deltaTime * v;
            v += deltaTime * xAcc;
            theta += deltaTime * omega;
            omega += deltaTime * thetaAcc;

            steps++;

            updateStateMatrix();

            // Terminate the episode if the cart goes out of bounds or the pole angle exceeds a threshold.
            if (x < -2.4f || x > 2.4f || theta < -0.209f || theta > 0.209f || steps >= maxSteps) {
                done = true;
            }
        }
    };

} 
