"""
🚀 OPTIMIZED RL TRAINING SCRIPT (v2.2 - Stability Fixes)
This script trains the model using batch processing for significant speed improvements.

✅ NEW: Instead of training step-by-step, this version collects a batch of
experiences (e.g., 256 steps) and sends the entire batch to the GPU for a
single, efficient, vectorized training update. This minimizes the CPU-GPU
communication overhead and better utilizes the GPU's parallel processing power.

🔧 FIX v2.2: Added a small epsilon to prevent ZeroDivisionError for instantaneous
episodes. The root cause of instability is addressed with gradient clipping
in the NeuralNetworkTraining_stabilized.py file.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import tensorflow as tf
import pandas as pd
import os

# Import the necessary components
from Suspension_Model import QuarterCarModel
from RewardFunction import StableSuspensionEnvironment
# IMPORTANT: Use the new stabilized training file
from NeuralNetworkArch import ActorCriticAgent

# --- Check for GPU ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("⚠️ No GPU detected. Training will run on the CPU.")


def run_training():
    print("🚀 RUNNING OPTIMIZED TRAINING WITH BATCH PROCESSING")
    print("===================================================")
    
    # --- Configuration ---
    # NOTE: A new directory is used to prevent overwriting original checkpoints.
    # If you wish to use old checkpoints, change this path to './training_checkpoints'
    checkpoint_dir = './training_checkpoints_batched'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    MAX_EPISODES = 10000
    STEPS_PER_EPISODE = 6000
    BATCH_SIZE = 256  # Number of experiences to collect before a training update
    LR_REDUCTION_EPISODE = 7000
    PRINT_EVERY = 10
    PLOT_EVERY = 100
    SAVE_EVERY = 100

    # 1. Initialize Components
    print("\n1️⃣  Initializing components...")
    car_model = QuarterCarModel()
    env = StableSuspensionEnvironment(car_model, dt=0.001)

    # 2. Initialize Agent
    print("\n2️⃣  Initializing Actor-Critic agent...")
    # Initialize with the stabilized agent class
    agent = ActorCriticAgent(actor_lr=0.001, critic_lr=0.01, gamma=0.99, action_bound=60.0)

    # 3. Setup Checkpoint Manager
    episode_counter = tf.Variable(0)
    checkpoint = tf.train.Checkpoint(
        episode_counter=episode_counter,
        actor=agent.actor,
        critic=agent.critic,
        actor_optimizer=agent.actor_optimizer,
        critic_optimizer=agent.critic_optimizer
    )
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

    # --- Restore from Checkpoint ---
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"✅ Restored from {manager.latest_checkpoint}")
        print(f"   Resuming training from episode {int(episode_counter)}...")
    else:
        print("🟡 No checkpoint found, starting from scratch.")

    # 4. The Training Loop (with batching and save/resume logic)
    print(f"\n4️⃣  Starting training... (Batch Size: {BATCH_SIZE})")
    print("-" * 50)
    
    episode_rewards = []
    running_avg_reward = deque(maxlen=100)
    
    # Store trajectories for plotting later
    if os.path.exists('training_rewards_batched.csv'):
        df = pd.read_csv('training_rewards_batched.csv')
        episode_rewards = df['reward'].tolist()
        running_avg_reward.extend(episode_rewards[-100:])

    try:
        # Start the loop from the restored episode number
        for episode in range(int(episode_counter), MAX_EPISODES):
            start_time = time.time()
            state = env.reset()
            episode_reward = 0
            
            # --- Batch Collection ---
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []

            for step in range(STEPS_PER_EPISODE):
                action, _ = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_states.append(next_state)
                batch_dones.append(done)

                state = next_state
                episode_reward += reward
                
                # If batch is full, train the agent
                if len(batch_states) >= BATCH_SIZE:
                    # Convert lists to tensors
                    state_tensor = tf.constant(batch_states, dtype=tf.float32)
                    action_tensor = tf.constant(batch_actions, dtype=tf.float32)
                    reward_tensor = tf.constant(batch_rewards, dtype=tf.float32)
                    next_state_tensor = tf.constant(batch_next_states, dtype=tf.float32)
                    done_tensor = tf.constant(batch_dones, dtype=tf.bool)
                    
                    # Perform a single, efficient, batched training step
                    agent.train_step(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)
                    
                    # Clear the batch
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

                if done:
                    break

            # --- Post-Episode ---
            episode_duration = time.time() - start_time
            # BUG FIX: Add a small epsilon to prevent division by zero
            steps_per_sec = STEPS_PER_EPISODE / (episode_duration + 1e-9)

            episode_rewards.append(episode_reward)
            running_avg_reward.append(episode_reward)
            episode_counter.assign_add(1)

            if episode % PRINT_EVERY == 0 or episode < int(episode_counter) + 5:
                print(f"Ep {int(episode_counter):<5} | Reward: {episode_reward:10.2f} | Avg(100): {np.mean(running_avg_reward):10.2f} | Steps/sec: {steps_per_sec:7.0f}")
        
            # Save checkpoint and plot progress
            if episode > 0 and episode % SAVE_EVERY == 0:
                manager.save()
                print(f"--- 💾 Checkpoint saved at episode {int(episode_counter)} ---")
                # Save rewards to csv
                pd.DataFrame({'reward': episode_rewards}).to_csv('training_rewards_batched.csv', index=False)

            if episode > 0 and episode % PLOT_EVERY == 0:
                plot_training_progress(episode_rewards)

            # Learning rate reduction
            if episode == LR_REDUCTION_EPISODE:
                print("--- 📉 Reducing learning rates ---")
                agent.actor_optimizer.learning_rate.assign(0.0001)
                agent.critic_optimizer.learning_rate.assign(0.001)

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user.")
    
    print("\n" + "=" * 50)
    print("🏁 TRAINING FINISHED")
    save_path = manager.save()
    print(f"💾 Final model state saved to: {save_path}")
    
    # Save final rewards
    pd.DataFrame({'reward': episode_rewards}).to_csv('training_rewards_batched.csv', index=False)
    plot_training_progress(episode_rewards, final=True)
    
    print("\n\nTo evaluate the latest checkpoint, run:")
    print("python evaluation_with_Passive.py")


def plot_training_progress(episode_rewards, final=False):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.6, color='cornflowerblue', label='Episode Reward')
    avg_series = pd.Series(episode_rewards).rolling(100, min_periods=10).mean()
    plt.plot(avg_series, color='red', linewidth=2, label='100-Episode Rolling Average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Optimized Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if final:
        plt.savefig("batched_training_progress.png")
        print("✅ Final training plot saved to 'batched_training_progress.png'")

    plt.draw()
    plt.pause(0.1)


if __name__ == "__main__":
    run_training()
    print("\nAll tasks complete. Close the plot windows to exit.")
    plt.show()
