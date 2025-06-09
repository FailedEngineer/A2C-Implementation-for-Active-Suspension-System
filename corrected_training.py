"""
🚀 CUSTOM ARCHITECTURE RL TRAINING SCRIPT
This script runs the training using your specific network architectures.

It imports the models from `NeuralNetworkTraining_Custom.py`.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from collections import deque

# Import the corrected environment and the CUSTOM agent
from Suspension_Model import QuarterCarModel
from RewardFunction import StableSuspensionEnvironment
from NeuralNetworkTraining import ActorCriticAgent # Using the new custom file

def run_training():
    """Main function to set up and run the training with custom architectures."""
    
    print("🚀 RUNNING TRAINING WITH CUSTOM ARCHITECTURES")
    print("===================================================")

    # 1. Initialize Components
    print("\n1️⃣  Initializing components...")
    car_model = QuarterCarModel()
    env = StableSuspensionEnvironment(car_model, dt=0.001)

    # 2. Initialize Agent with CUSTOM Architectures
    print("\n2️⃣  Initializing Actor-Critic agent with YOUR specified architectures...")
    agent = ActorCriticAgent(
        actor_lr=0.001,
        critic_lr=0.01,
        gamma=0.99,
        action_bound=60.0
    )

    # 3. Setup Training Parameters
    MAX_EPISODES = 2000
    STEPS_PER_EPISODE = 3000
    LR_REDUCTION_EPISODE = 500
    PRINT_EVERY = 20
    PLOT_EVERY = 100

    print(f"\n3️⃣  Training parameters:")
    print(f"   Max Episodes: {MAX_EPISODES}")
    print(f"   Steps per Episode: {STEPS_PER_EPISODE}")

    # 4. The Training Loop
    print("\n4️⃣  Starting training...")
    print("-" * 50)

    episode_rewards = []
    running_avg_reward = deque(maxlen=100)
    start_time = time.time()

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for step in range(STEPS_PER_EPISODE):
            action, log_prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Convert to tensors for training
            state_tensor = tf.constant([[state]], dtype=tf.float32)
            action_tensor = tf.constant([[action]], dtype=tf.float32)
            reward_tensor = tf.constant([[reward]], dtype=tf.float32)
            next_state_tensor = tf.constant([[next_state]], dtype=tf.float32)
            done_tensor = tf.constant([[done]], dtype=tf.bool)
            
            agent.train_step(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
    
        episode_rewards.append(episode_reward)
        running_avg_reward.append(episode_reward)
        current_avg = np.mean(running_avg_reward)
        
        if episode == LR_REDUCTION_EPISODE:
            agent.actor_optimizer.learning_rate.assign(0.0001)
            agent.critic_optimizer.learning_rate.assign(0.001)
            print(f"\n📉 Reduced learning rates at episode {episode}")
    
        if episode % PRINT_EVERY == 0 or episode < 5:
            elapsed = time.time() - start_time
            print(f"Ep {episode:4d} | Reward: {episode_reward:9.1f} | Avg(100): {current_avg:9.1f} | Time: {elapsed:.1f}s")
    
        if episode > 0 and episode % PLOT_EVERY == 0:
            plot_progress(episode_rewards)

    print("\n" + "=" * 50)
    print("🏁 TRAINING COMPLETE")
    plot_progress(episode_rewards, final=True)

def plot_progress(episode_rewards, final=False):
    """Helper function to plot training progress."""
    plt.figure(figsize=(12, 5))
    plt.plot(episode_rewards, alpha=0.5, color='cornflowerblue', label='Episode Reward')
    
    avg_series = pd.Series(episode_rewards).rolling(100, min_periods=1).mean()
    plt.plot(avg_series, color='red', linewidth=2, label='100-Episode Average')
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress (Custom Architecture)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if final:
        plt.savefig("custom_arch_training_progress.png")
    plt.show()

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not found. Please install it: pip install pandas")
        exit()
        
    run_training()
