"""
🔬 AGENT EVALUATION SCRIPT
This script loads a trained agent from a checkpoint and evaluates its performance
on the two test scenarios described in the research paper.

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

# --- Evaluation Environment for Bumpy Road ---
class BumpyRoadEnvironment(StableSuspensionEnvironment):
    """
    An evaluation environment that generates a bumpy road profile,
    similar to Figure 8a in the research paper.
    """
    def __init__(self, car_model, dt=0.001):
        super().__init__(car_model, dt)
        self.noise_amp = 0.015
        self.base_amp = 0.025
        self.period = 30.0 # Long period for the base wave

    def get_smooth_road_profile(self, t):
        """Generates a pseudo-random bumpy road profile."""
        base_profile = self.base_amp * (1 + np.sin(2 * np.pi * t / self.period)) / 2
        if not hasattr(self, 'noise') or len(self.noise) < int(self.time/self.dt) + 2:
            self.noise = (np.random.rand(10000) - 0.5) * self.noise_amp
            self.smooth_noise = pd.Series(self.noise).rolling(window=50, min_periods=1).mean().to_numpy()

        idx = int(t / self.dt) % len(self.smooth_noise)
        zr = base_profile + self.smooth_noise[idx]
        zr_dot = (zr - self.prev_zr) / self.dt
        self.prev_zr = zr
        return zr, zr_dot

# --- Plotting for Evaluation ---
def plot_evaluation_results(sq_results, bumpy_results):
    """Plots the evaluation results in a format similar to the paper."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Trained Agent Performance Evaluation', fontsize=16)

    axs[0, 0].set_title('Scenario 1: Square Wave Road')
    axs[0, 0].plot(sq_results['time'], sq_results['xs'], label='Sprung Mass (Body)')
    axs[0, 0].set_ylabel('Displacement (m)'); axs[0, 0].grid(True); axs[0, 0].legend()
    axs[1, 0].plot(sq_results['time'], sq_results['xus'], 'g', label='Unsprung Mass (Wheel)')
    axs[1, 0].set_ylabel('Displacement (m)'); axs[1, 0].grid(True); axs[1, 0].legend()
    axs[2, 0].plot(sq_results['time'], sq_results['accel'], 'r', label='Body Acceleration')
    axs[2, 0].set_ylabel('Acceleration (m/s²)')
    axs[2, 0].set_xlabel('Time (s)'); axs[2, 0].grid(True); axs[2, 0].legend()

    axs[0, 1].set_title('Scenario 2: Bumpy Road')
    axs[0, 1].plot(bumpy_results['time'], bumpy_results['xs']); axs[0, 1].grid(True)
    axs[1, 1].plot(bumpy_results['time'], bumpy_results['xus'], 'g'); axs[1, 1].grid(True)
    axs[2, 1].plot(bumpy_results['time'], bumpy_results['accel'], 'r')
    axs[2, 1].set_xlabel('Time (s)'); axs[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("agent_evaluation_results.png")
    plt.show() # Block here to show the final plots

# --- Main Evaluation Function ---
def evaluate_agent(checkpoint_path):
    """
    Loads a agent from a checkpoint and tests it.
    """
    print("\n" + "="*50)
    print("🔬 EVALUATING TRAINED AGENT...")
    print(f"   Loading from: {checkpoint_path}")
    print("="*50)

    # 1. Initialize a blank agent
    agent = ActorCriticAgent(actor_lr=0.001, critic_lr=0.01, gamma=0.99, action_bound=60.0)

    # 2. Create a checkpoint object that mirrors the one from training
    # We only need to restore the actor and critic for evaluation.
    checkpoint = tf.train.Checkpoint(
        actor=agent.actor,
        critic=agent.critic
    )

    # 3. Restore the weights from the specified checkpoint file
    # Using expect_partial to ignore optimizer state and episode counter.
    status = checkpoint.restore(checkpoint_path).expect_partial()
    print("✅ Agent state restored.")

    # --- Scenario 1: Square Wave Test ---
    print("\n1. Testing on Square Wave Road Profile...")
    STEPS_PER_EPISODE = 6000 # Use a fixed length for evaluation
    eval_env_sq = StableSuspensionEnvironment(QuarterCarModel())
    state = eval_env_sq.reset()
    sq_results = {'time': [], 'xs': [], 'xus': [], 'accel': []}

    for step in range(STEPS_PER_EPISODE):
        mu, _ = agent.actor(tf.constant([[state]], dtype=tf.float32))
        action = mu.numpy()[0, 0]
        next_state, _, _, info = eval_env_sq.step(action)
        state = next_state
        sq_results['time'].append(info['time'])
        sq_results['xs'].append(eval_env_sq.car_model.state[0])
        sq_results['xus'].append(eval_env_sq.car_model.state[2])
        sq_results['accel'].append(info['body_acceleration'])

    # --- Scenario 2: Bumpy Road Test ---
    print("\n2. Testing on Bumpy Road Profile...")
    eval_env_bumpy = BumpyRoadEnvironment(QuarterCarModel())
    state = eval_env_bumpy.reset()
    bumpy_results = {'time': [], 'xs': [], 'xus': [], 'accel': []}

    for step in range(STEPS_PER_EPISODE):
        mu, _ = agent.actor(tf.constant([[state]], dtype=tf.float32))
        action = mu.numpy()[0, 0]
        next_state, _, _, info = eval_env_bumpy.step(action)
        state = next_state
        bumpy_results['time'].append(info['time'])
        bumpy_results['xs'].append(eval_env_bumpy.car_model.state[0])
        bumpy_results['xus'].append(eval_env_bumpy.car_model.state[2])
        bumpy_results['accel'].append(info['body_acceleration'])
        
    plot_evaluation_results(sq_results, bumpy_results)
    print("\nEvaluation complete. Plot saved to agent_evaluation_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained Actor-Critic agent.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to a specific checkpoint file (e.g., ./training_checkpoints/ckpt-10). If not provided, the latest checkpoint will be used.')
    args = parser.parse_args()

    checkpoint_dir = './training_checkpoints'

    if args.checkpoint_path:
        checkpoint_to_load = args.checkpoint_path
    else:
        # Find the latest checkpoint in the directory
        checkpoint_to_load = tf.train.latest_checkpoint(checkpoint_dir)

    if not checkpoint_to_load:
        print("❌ Error: No checkpoint found.")
        print("Please train a model first using `run_custom_arch_training.py` or provide a path using --checkpoint_path.")
    else:
        evaluate_agent(checkpoint_to_load)
