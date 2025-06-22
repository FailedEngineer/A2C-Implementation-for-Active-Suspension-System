"""
🔬 AGENT EVALUATION SCRIPT (ENHANCED)
This script loads a trained agent from a checkpoint and evaluates its performance
against a passive system on the two test scenarios from the research paper.

How to run:
1. To test the latest checkpoint:
   python evaluate_agent.py

2. To test a specific checkpoint:
   python evaluate_agent.py --checkpoint_path ./training_checkpoints/ckpt-10
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import argparse

# Import necessary components from other project files
from Suspension_Model import QuarterCarModel
from RewardFunction import StableSuspensionEnvironment
from NeuralNetworkTraining import ActorCriticAgent

# --- Evaluation Environment for Square Wave Road ---
class SquareWaveEnvironment(StableSuspensionEnvironment):
    """
    An evaluation environment that generates a square wave road profile,
    as described in Figure 6a of the research paper. This is used for Scenario 1 testing.
    """
    def __init__(self, car_model, dt=0.001):
        super().__init__(car_model, dt)
        # Parameters from the paper for the square wave
        self.road_amplitude = 0.02
        self.road_period = 3.0

    def get_smooth_road_profile(self, t):
        """
        Generates a road profile with smooth transitions (ramps) to be more realistic
        than an instantaneous step, which prevents infinite derivatives (zr_dot).
        """
        cycle_time = t % self.road_period
        half_period = self.road_period / 2
        transition_duration = 0.01 # Use a small 10ms ramp time for the transition

        zr = 0.0
        # Ramp up at the beginning of the cycle
        if cycle_time < transition_duration:
            progress = cycle_time / transition_duration
            zr = self.road_amplitude * progress
        # Flat top section
        elif cycle_time < half_period - transition_duration:
            zr = self.road_amplitude
        # Ramp down in the middle of the cycle
        elif cycle_time < half_period + transition_duration:
            progress = (cycle_time - (half_period - transition_duration)) / transition_duration
            zr = self.road_amplitude * (1.0 - progress)
        # Flat bottom section (zero height)
        else:
            zr = 0.0

        # Calculate derivative numerically, which is now stable due to the ramps
        zr_dot = (zr - self.prev_zr) / self.dt
        self.prev_zr = zr
        return zr, zr_dot


# --- Evaluation Environment for Bumpy Road ---
class BumpyRoadEnvironment(StableSuspensionEnvironment):
    """
    An evaluation environment that generates a bumpy road profile,
    similar to Figure 8a in the research paper. This is used for Scenario 2 testing.
    """
    def __init__(self, car_model, dt=0.001):
        super().__init__(car_model, dt)
        self.noise_amp = 0.015
        self.base_amp = 0.025
        self.period = 30.0 # Long period for the base wave

    def get_smooth_road_profile(self, t):
        """Generates a pseudo-random bumpy road profile."""
        base_profile = self.base_amp * (1 + np.sin(2 * np.pi * t / self.period)) / 2
        # Generate repeatable noise for consistent evaluation runs
        if not hasattr(self, 'noise') or len(self.noise) < int(self.time/self.dt) + 2:
            np.random.seed(42) # Use a fixed seed for reproducibility
            noise = (np.random.rand(10000) - 0.5) * self.noise_amp
            # Smooth the noise to make it more realistic
            self.smooth_noise = pd.Series(noise).rolling(window=50, min_periods=1).mean().to_numpy()

        idx = int(t / self.dt) % len(self.smooth_noise)
        zr = base_profile + self.smooth_noise[idx]
        zr_dot = (zr - self.prev_zr) / self.dt
        self.prev_zr = zr
        return zr, zr_dot

# --- Plotting for Evaluation ---
def plot_evaluation_results(sq_results, bumpy_results, passive_sq_results, passive_bumpy_results):
    """Plots the evaluation results, comparing the agent to a passive system."""
    fig, axs = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
    fig.suptitle('Agent vs. Passive System Performance Evaluation', fontsize=18, fontweight='bold')

    # --- Scenario 1: Square Wave Road ---
    axs[0, 0].set_title('Scenario 1: Square Wave Road', fontsize=14)
    axs[0, 0].plot(sq_results['time'], sq_results['accel'], 'r-', linewidth=2, label='Agent Body Accel.')
    axs[0, 0].plot(passive_sq_results['time'], passive_sq_results['accel'], 'k--', label='Passive Body Accel.')
    axs[0, 0].set_ylabel('Acceleration (m/s²)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

    axs[1, 0].plot(sq_results['time'], sq_results['susp_travel'], 'b-', linewidth=2, label='Agent Susp. Travel')
    axs[1, 0].plot(passive_sq_results['time'], passive_sq_results['susp_travel'], 'k--', label='Passive Susp. Travel')
    axs[1, 0].set_ylabel('Susp. Travel (m)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

    axs[2, 0].plot(sq_results['time'], sq_results['force'], 'g-', linewidth=2, label='Agent Control Force')
    axs[2, 0].set_ylabel('Force (N)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].legend()
    axs[2, 0].grid(True, which='both', linestyle='--', linewidth=0.5)


    # --- Scenario 2: Bumpy Road ---
    axs[0, 1].set_title('Scenario 2: Bumpy Road', fontsize=14)
    axs[0, 1].plot(bumpy_results['time'], bumpy_results['accel'], 'r-', linewidth=2, label='Agent Body Accel.')
    axs[0, 1].plot(passive_bumpy_results['time'], passive_bumpy_results['accel'], 'k--', label='Passive Body Accel.')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

    axs[1, 1].plot(bumpy_results['time'], bumpy_results['susp_travel'], 'b-', linewidth=2, label='Agent Susp. Travel')
    axs[1, 1].plot(passive_bumpy_results['time'], passive_bumpy_results['susp_travel'], 'k--', label='Passive Susp. Travel')
    axs[1, 1].legend()
    axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

    axs[2, 1].plot(bumpy_results['time'], bumpy_results['force'], 'g-', linewidth=2, label='Agent Control Force')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].legend()
    axs[2, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("agent_vs_passive_evaluation.png", dpi=300)
    plt.show()

def run_simulation(env, agent=None, steps=6000):
    """Runs a simulation for a given environment and agent (or passive)."""
    state = env.reset()
    results = {'time': [], 'xs': [], 'xus': [], 'susp_travel': [], 'accel': [], 'force': []}

    for step in range(steps):
        if agent:
            # For evaluation, use the deterministic action (mean of the distribution) for stability
            mu, _ = agent.actor(tf.constant([[state]], dtype=tf.float32))
            action = mu.numpy()[0, 0]
        else:
            # Passive system, no control force applied
            action = 0.0

        next_state, _, _, info = env.step(action)
        state = next_state

        results['time'].append(info['time'])
        results['xs'].append(env.car_model.state[0])
        results['xus'].append(env.car_model.state[2])
        results['susp_travel'].append(info['suspension_travel'])
        results['accel'].append(info['body_acceleration'])
        results['force'].append(action)

    return results

# --- Main Evaluation Function ---
def evaluate_agent(checkpoint_path):
    """
    Loads an agent from a checkpoint and tests it against a passive system.
    """
    print("\n" + "="*50)
    print("🔬 EVALUATING TRAINED AGENT...")
    print(f"   Loading weights from: {checkpoint_path}")
    print("="*50)

    # 1. Initialize a blank agent
    agent = ActorCriticAgent()

    # 2. Create a checkpoint object that mirrors the one from training
    # We only need to restore the actor and critic networks for evaluation.
    checkpoint = tf.train.Checkpoint(actor=agent.actor, critic=agent.critic)

    # 3. Restore the weights from the specified checkpoint file
    # Using .expect_partial() ignores optimizer state and counters, which is fine for evaluation.
    status = checkpoint.restore(checkpoint_path).expect_partial()
    print("✅ Agent weights restored successfully.")
    status.assert_existing_objects_matched() # Confirms that the actor/critic weights were loaded.


    # --- Run Simulations for Both Agent and Passive System ---

    # Scenario 1: Square Wave Test
    print("\n1. Running Scenario 1: Square Wave Road Profile...")
    sq_env = SquareWaveEnvironment(QuarterCarModel())
    agent_sq_results = run_simulation(sq_env, agent=agent)

    sq_env_passive = SquareWaveEnvironment(QuarterCarModel())
    passive_sq_results = run_simulation(sq_env_passive, agent=None) # No agent for passive

    # Scenario 2: Bumpy Road Test
    print("\n2. Running Scenario 2: Bumpy Road Profile...")
    bumpy_env = BumpyRoadEnvironment(QuarterCarModel())
    agent_bumpy_results = run_simulation(bumpy_env, agent=agent)

    bumpy_env_passive = BumpyRoadEnvironment(QuarterCarModel())
    passive_bumpy_results = run_simulation(bumpy_env_passive, agent=None) # No agent for passive


    # --- Plotting the comparative results ---
    print("\n3. Generating comparison plots...")
    plot_evaluation_results(agent_sq_results, agent_bumpy_results, passive_sq_results, passive_bumpy_results)
    print("\nEvaluation complete. Plot saved to 'agent_vs_passive_evaluation.png'")

if __name__ == "__main__":
    # Set up argument parser to allow specifying a checkpoint
    parser = argparse.ArgumentParser(description='Evaluate a trained Actor-Critic agent for active suspension.')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help='Path to a specific checkpoint file (e.g., ./training_checkpoints/ckpt-2000). If not provided, the latest checkpoint will be used.'
    )
    args = parser.parse_args()

    checkpoint_dir = './training_checkpoints_batched_v2'

    if args.checkpoint_path:
        checkpoint_to_load = args.checkpoint_path
    else:
        # If no path is given, find the latest checkpoint in the directory
        print("No specific checkpoint path provided. Searching for the latest checkpoint...")
        checkpoint_to_load = tf.train.latest_checkpoint(checkpoint_dir)

    if not checkpoint_to_load:
        print("\n❌ Error: No checkpoint found.")
        print("Please train a model first using 'corrected_training.py' or provide a valid path using the --checkpoint_path argument.")
    else:
        evaluate_agent(checkpoint_to_load)
