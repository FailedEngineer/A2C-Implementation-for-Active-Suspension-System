# run_this_now.py
"""
🚀 COMPLETE WORKING SOLUTION
This script fixes all issues and runs proper training.
Just run: python run_this_now.py
"""

import numpy as np
import matplotlib.pyplot as plt
from Suspension_Model import QuarterCarModel
from NeuralNetworkTraining import ActorCriticAgent
from RewardFunction import RewardFunction, SuspensionEnvironment
import time

print("🚀 ACTIVE SUSPENSION RL TRAINING - FIXED VERSION")
print("=" * 70)

# 1. Initialize with CORRECT parameters
print("\n1️⃣ Initializing components with CORRECT parameters...")
car_model = QuarterCarModel()
reward_func = RewardFunction()  # Uses k1=1000, k2=0.1 by default

# Verify parameters
print(f"   Reward function: k1={reward_func.k1}, k2={reward_func.k2}")
if reward_func.k1 != 1000 or reward_func.k2 != 0.1:
    print("   ❌ ERROR: Wrong reward parameters! Fixing...")
    reward_func.k1 = 1000
    reward_func.k2 = 0.1

env = SuspensionEnvironment(car_model, reward_func)

# 2. Initialize agent with paper's learning rates
print("\n2️⃣ Initializing Actor-Critic agent...")
agent = ActorCriticAgent(
    actor_lr=0.001,   # Paper's initial value
    critic_lr=0.01,   # 10x actor rate
    gamma=0.99,
    action_bound=60.0
)
print(f"   Actor LR: {agent.actor_optimizer.learning_rate.numpy()}")
print(f"   Critic LR: {agent.critic_optimizer.learning_rate.numpy()}")

# 3. Training parameters
MAX_EPISODES = 2000
STEPS_PER_EPISODE = 1500  # Paper: 1.5 seconds at dt=0.001
LR_REDUCTION_EPISODE = 500
PRINT_EVERY = 50
PLOT_EVERY = 200

print(f"\n3️⃣ Training parameters:")
print(f"   Episodes: {MAX_EPISODES}")
print(f"   Steps per episode: {STEPS_PER_EPISODE}")
print(f"   LR reduction at: episode {LR_REDUCTION_EPISODE}")

# 4. Training loop
print("\n4️⃣ Starting training...")
print("   Expected initial rewards: -15,000 to -20,000")
print("   Expected final rewards: -10,000 to -12,000")
print("   (NOT -50! That would be wrong!)")
print("-" * 70)

episode_rewards = []
running_avg = []
start_time = time.time()

for episode in range(MAX_EPISODES):
    # Reset environment
    state = env.reset()
    episode_reward = 0
    
    # Episode loop
    for step in range(STEPS_PER_EPISODE):
        # Get action
        action, log_prob = agent.get_action(state)
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # Train agent
        actor_loss, critic_loss, td_error = agent.train_step(
            state, action, reward, next_state, done
        )
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    # Record episode reward
    episode_rewards.append(episode_reward)
    
    # Calculate running average
    window = min(50, len(episode_rewards))
    if len(episode_rewards) >= window:
        avg = np.mean(episode_rewards[-window:])
        running_avg.append(avg)
    else:
        running_avg.append(episode_reward)
    
    # Reduce learning rates at specified episode
    if episode == LR_REDUCTION_EPISODE:
        agent.actor_optimizer.learning_rate.assign(0.0001)
        agent.critic_optimizer.learning_rate.assign(0.001)
        print(f"\n📉 Reduced learning rates at episode {episode}")
        print(f"   Actor: 0.001 → 0.0001")
        print(f"   Critic: 0.01 → 0.001")
    
    # Print progress
    if episode % PRINT_EVERY == 0 or episode < 5:
        elapsed = time.time() - start_time
        print(f"Episode {episode:4d} | "
              f"Reward: {episode_reward:8.1f} | "
              f"Avg: {running_avg[-1]:8.1f} | "
              f"Time: {elapsed:.1f}s")
    
    # Plot progress
    if episode % PLOT_EVERY == 0 and episode > 0:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, 'b-', alpha=0.3, label='Episode rewards')
        plt.plot(running_avg, 'r-', linewidth=2, label='Running average')
        plt.axhline(y=-10328, color='g', linestyle='--', label='Paper best: -10,328')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        recent = episode_rewards[-200:] if len(episode_rewards) > 200 else episode_rewards
        plt.hist(recent, bins=30, alpha=0.7, color='blue')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Recent Episode Rewards Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Check for convergence
    if len(running_avg) >= 100 and running_avg[-1] > -12000:
        print(f"\n✅ Converged at episode {episode}!")
        print(f"   Final average reward: {running_avg[-1]:.1f}")
        break

# 5. Final evaluation
print("\n5️⃣ Training complete! Evaluating final performance...")

# Test on 10 episodes
test_rewards = []
for _ in range(10):
    state = env.reset()
    episode_reward = 0
    
    for step in range(STEPS_PER_EPISODE):
        # Use mean action (deterministic)
        mu, std = agent.actor(np.array([[state]], dtype=np.float32))
        action = float(mu.numpy()[0, 0])
        action = np.clip(action, -60, 60)
        
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    test_rewards.append(episode_reward)

# 6. Final results
print("\n" + "=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)
print(f"Training episodes: {len(episode_rewards)}")
print(f"Final training average: {running_avg[-1]:.1f}")
print(f"Test performance: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
print(f"Paper's best result: -10,328")
print(f"Your improvement: {((running_avg[-1] + 10328) / 10328 * 100):.1f}% from paper's best")

# 7. Final plot
plt.figure(figsize=(15, 10))

# Training progress
plt.subplot(2, 2, 1)
plt.plot(episode_rewards, 'b-', alpha=0.5, linewidth=0.5)
plt.plot(running_avg, 'r-', linewidth=2)
plt.axhline(y=-10328, color='g', linestyle='--', label='Paper: -10,328')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Complete Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

# Learning curve zoom
plt.subplot(2, 2, 2)
if len(episode_rewards) > 100:
    plt.plot(episode_rewards[50:], 'b-', alpha=0.5, linewidth=0.5)
    plt.plot(range(50, len(running_avg)), running_avg[50:], 'r-', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Progress (after initial exploration)')
plt.grid(True, alpha=0.3)

# Reward distribution
plt.subplot(2, 2, 3)
plt.hist(episode_rewards, bins=50, alpha=0.7, color='blue')
plt.xlabel('Episode Reward')
plt.ylabel('Frequency')
plt.title('Episode Reward Distribution')
plt.grid(True, alpha=0.3)

# Test performance
plt.subplot(2, 2, 4)
plt.bar(range(len(test_rewards)), test_rewards, color='green', alpha=0.7)
plt.axhline(y=np.mean(test_rewards), color='red', linestyle='--', 
            label=f'Mean: {np.mean(test_rewards):.1f}')
plt.xlabel('Test Episode')
plt.ylabel('Reward')
plt.title('Test Performance (10 episodes)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Training complete! Check the plots above.")
print("🎯 If your rewards are around -10,000 to -12,000, you've succeeded!")
print("❌ If your rewards are around -50 to -200, something is still wrong with the reward function.")