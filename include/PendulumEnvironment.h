#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include "Environment.h"
#include "Matrix.h"

namespace BaseML::RL::Tests {

    class PendulumEnvironment : public Environment {
    private:
        static constexpr float PI = 3.14159265359f;

        // Only one player (agent)
        std::vector<std::string> players = { "agent" };

        // Pendulum dynamics parameters
        const float g = 9.81f;    // gravitational acceleration (m/s^2)
        const float l = 1.0f;     // pendulum length (m)
        const float m = 1.0f;     // mass (kg)
        // Moment of inertia I = m * l^2 for a simple pendulum
        const float I = m * l * l;

        // State variables: theta (angle in radians) and theta_dot (angular velocity)
        float theta, theta_dot;
        // Last applied torque (action)
        float torque;

        // Simulation step counter and episode length limit
        int steps;
        const int maxSteps = 200;

        bool done;
        bool initialized;

        // State represented as a column vector: [cos(theta), sin(theta), theta_dot]^T
        Matrix stateMatrix;

        // Helper to update the stateMatrix based on current theta and theta_dot
        void updateStateMatrix() {
            stateMatrix = Matrix({ std::cos(theta), std::sin(theta), theta_dot }, true);
        }

    public:
        // Observation dimension is 3, action dimension is 1.
        PendulumEnvironment()
            : Environment(3, 1),
            theta(0.0f), theta_dot(0.0f), torque(0.0f),
            steps(0), done(false), initialized(false),
            stateMatrix({ 1.0f, 0.0f, 0.0f }, true) // initial state: cos(0)=1, sin(0)=0, theta_dot=0
        {
        }

        // Return the list of players (here, only one agent)
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

        // Close the environment (here simply mark as uninitialized)
        void close() override {
            initialized = false;
        }

        // Reset the environment to an initial state.
        // Here we optionally randomize the initial angle and angular velocity.
        void reset() override {
            theta = ((float)std::rand() / RAND_MAX) * 0.4f - 0.2f;     // random in [-0.2, 0.2] radians
            theta_dot = ((float)std::rand() / RAND_MAX) * 0.4f - 0.2f;   // random in [-0.2, 0.2] rad/s
            torque = 0.0f;
            steps = 0;
            done = false;
            updateStateMatrix();
        }

        // Terminate the episode when maxSteps is reached.
        bool isFinished() override {
            return done;
        }

        // Render the current state to the console.
        void render() override {
            std::cout << "Step: " << steps
                << ", theta: " << theta
                << ", theta_dot: " << theta_dot
                << ", torque: " << torque << std::endl;
        }

        // Return the current state as a column Matrix.
        const Matrix& getState(const char* playerId) const override {
            return stateMatrix;
        }

        // Set the action (torque) for the agent.
        // Expects a 1x1 Matrix where the first element is the applied torque.
        void setAction(const char* playerId, const Matrix& action) override {
            torque = action(0);
        }

        // Compute the reward.
        // Here we use a cost function: theta^2 + 0.1*theta_dot^2 + 0.001*torque^2.
        // The reward is defined as the negative cost, so higher reward is achieved when
        // the pendulum is close to the upright position (theta near 0) and with small angular velocity.
        float getReward(const char* playerId) const override {
            float cost = theta * theta + 0.1f * theta_dot * theta_dot + 0.001f * torque * torque;
            return -cost;
        }

        // Update the environment's state by deltaTime seconds.
        void update(float deltaTime) override {
            if (done)
                return;

            // Compute angular acceleration: theta_ddot = - (g/l)*sin(theta) + torque / I.
            float theta_ddot = -(g / l) * std::sin(theta) + torque / I;

            // Update angular velocity and angle using Euler integration.
            theta_dot += deltaTime * theta_ddot;
            theta += deltaTime * theta_dot;

            // Optionally wrap theta within [-pi, pi] for numerical stability.
            if (theta > PI)
                theta -= 2 * PI;
            else if (theta < -PI)
                theta += 2 * PI;

            steps++;
            updateStateMatrix();

            if (steps >= maxSteps)
                done = true;
        }
    };

} // namespace BaseML::RL