"""
🚀 CUSTOM ARCHITECTURE RL TRAINING SCRIPT
This script trains the model and saves checkpoints.

✅ NEW: Now focused only on training. Evaluation has been moved to `evaluate_agent.py`.
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
from NeuralNetworkTraining import ActorCriticAgent

def run_training():
    print("🚀 RUNNING TRAINING WITH CUSTOM ARCHITECTURES")
    print("===================================================")
    
    # --- Checkpoint Configuration ---
    checkpoint_dir = './training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Initialize Components
    print("\n1️⃣  Initializing components...")
    car_model = QuarterCarModel()
    env = StableSuspensionEnvironment(car_model, dt=0.001)

    # 2. Initialize Agent with CUSTOM Architectures
    print("\n2️⃣  Initializing Actor-Critic agent...")
    agent = ActorCriticAgent(actor_lr=0.001, critic_lr=0.01, gamma=0.99, action_bound=60.0)

    # --- Setup Checkpoint Manager ---
    episode_counter = tf.Variable(0)
    checkpoint = tf.train.Checkpoint(
        episode_counter=episode_counter,
        actor=agent.actor,
        critic=agent.critic,
        actor_optimizer=agent.actor_optimizer,
        critic_optimizer=agent.critic_optimizer
    )
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # --- Restore from Checkpoint ---
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"✅ Restored from {manager.latest_checkpoint}")
        print(f"   Resuming training from episode {int(episode_counter)}...")
    else:
        print("🟡 No checkpoint found, starting from scratch.")

    # 3. Setup Training Parameters
    MAX_EPISODES = 2000
    STEPS_PER_EPISODE = 3000
    LR_REDUCTION_EPISODE = 500
    PRINT_EVERY = 20
    PLOT_EVERY = 100

    # 4. The Training Loop (with save/resume logic)
    print("\n4️⃣  Starting training... (Press Ctrl+C to interrupt and save)")
    print("-" * 50)
    episode_rewards = []
    running_avg_reward = deque(maxlen=100)
    
    try:
        # Start the loop from the restored episode number
        for episode in range(int(episode_counter), MAX_EPISODES):
            state = env.reset()
            episode_reward = 0
            
            for step in range(STEPS_PER_EPISODE):
                action, log_prob = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                tensors = [tf.constant([[d]], dtype=tf.float32) for d in [state, action, reward, next_state]]
                done_tensor = tf.constant([[done]], dtype=tf.bool)
                agent.train_step(tensors[0], tensors[1], tensors[2], tensors[3], done_tensor)
                
                episode_reward += reward
                state = next_state
                if done: break
        
            episode_rewards.append(episode_reward)
            running_avg_reward.append(episode_reward)
            episode_counter.assign_add(1) # Increment episode counter

            if episode % PRINT_EVERY == 0 or episode < int(episode_counter) + 5:
                print(f"Ep {episode:4d} | Reward: {episode_reward:9.1f} | Avg(100): {np.mean(running_avg_reward):9.1f}")
        
            if episode > 0 and episode % PLOT_EVERY == 0:
                plot_training_progress(episode_rewards)
                manager.save()
                print(f"--- 💾 Checkpoint saved at episode {episode} ---")
            
            if episode == LR_REDUCTION_EPISODE:
                agent.actor_optimizer.learning_rate.assign(0.0001)
                agent.critic_optimizer.learning_rate.assign(0.001)

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user.")
    
    print("\n" + "=" * 50)
    print("🏁 TRAINING FINISHED")
    save_path = manager.save()
    print(f"💾 Final model state saved to: {save_path}")
    
    plot_training_progress(episode_rewards, final=True)
    
    # --- MODIFIED: Removed direct evaluation call ---
    print("\n\nTo evaluate the latest checkpoint, run:")
    print("python evaluate_agent.py")
    print("\nTo evaluate a specific checkpoint, run (for example):")
    print(f"python evaluate_agent.py --checkpoint_path {save_path}")


def plot_training_progress(episode_rewards, final=False):
    plt.figure(figsize=(12, 5))
    plt.plot(episode_rewards, alpha=0.5, color='cornflowerblue', label='Episode Reward')
    avg_series = pd.Series(episode_rewards).rolling(100, min_periods=1).mean()
    plt.plot(avg_series, color='red', linewidth=2, label='100-Episode Average')
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.title("Training Progress (Custom Architecture)"); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    if final:
        plt.savefig("custom_arch_training_progress.png")

    plt.draw()
    plt.pause(0.1)

if __name__ == "__main__":
    run_training()
    
    print("\nAll tasks complete. Training plots are displayed.")
    print("Close the plot windows to exit the script.")
    plt.show()
