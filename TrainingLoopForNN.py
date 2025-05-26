import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from collections import deque
import pandas as pd

class TrainingLoop:
    """
    Training loop for TD Advantage Actor-Critic on Active Suspension Control.
    Based on the paper: "Online Reinforcement Learning-Based Control of 
    an Active Suspension System Using the Actor Critic Approach"
    """
    
    def __init__(self, env, agent, max_episodes=2000, max_steps_per_episode=10000):
        """
        Initialize training loop.
        
        Parameters:
        -----------
        env : SuspensionEnvironment
            The suspension environment
        agent : ActorCriticAgent
            The actor-critic agent
        max_episodes : int
            Maximum training episodes
        max_steps_per_episode : int
            Maximum steps per episode (10 seconds at dt=0.001)
        """
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.running_reward = deque(maxlen=100)
        
        # Performance tracking
        self.best_avg_reward = -np.inf
        self.convergence_threshold = -50  # Much better than random (-113 average)
        
    def train(self, verbose=True, plot_every=100, save_every=500):
        """
        Main training loop implementing TD Advantage Actor-Critic.
        
        Parameters:
        -----------
        verbose : bool
            Print training progress
        plot_every : int
            Plot progress every N episodes
        save_every : int
            Save model every N episodes
            
        Returns:
        --------
        training_stats : dict
            Training statistics and metrics
        """
        if verbose:
            print("🚀 Starting TD Actor-Critic Training...")
            print(f"📊 Target: Beat random baseline (-113.35 avg reward)")
            print(f"🎯 Success threshold: {self.convergence_threshold} avg reward")
            print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            episode_start = time.time()
            
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_actor_losses = []
            episode_critic_losses = []
            
            for step in range(self.max_steps_per_episode):
                # Get action from agent
                action, log_prob = self.agent.get_action(state)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Train the agent (online learning)
                actor_loss, critic_loss, td_error = self.agent.train_step(
                    state, action, reward, next_state, done
                )
                
                # Track metrics
                episode_reward += reward
                episode_actor_losses.append(actor_loss)
                episode_critic_losses.append(critic_loss)
                
                # Update state
                state = next_state
                
                # Check termination conditions
                if done:
                    break
                    
                # Early stopping for very poor performance (runaway)
                if step > 1000 and episode_reward < -10000:
                    if verbose and episode % 100 == 0:
                        print(f"  Early stopping episode {episode} (runaway detected)")
                    break
            
            # Episode finished - record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            self.running_reward.append(episode_reward)
            
            if len(episode_actor_losses) > 0:
                self.actor_losses.append(np.mean(episode_actor_losses))
                self.critic_losses.append(np.mean(episode_critic_losses))
            else:
                self.actor_losses.append(0)
                self.critic_losses.append(0)
            
            # Calculate running average
            avg_reward = np.mean(self.running_reward)
            
            # Track best performance
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
            
            # Print progress
            if verbose and (episode % 50 == 0 or episode < 10):
                episode_time = time.time() - episode_start
                elapsed = time.time() - start_time
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.1f} | "
                      f"Avg(100): {avg_reward:7.1f} | "
                      f"Steps: {step+1:4d} | "
                      f"Time: {episode_time:.2f}s | "
                      f"Elapsed: {elapsed/60:.1f}m")
            
            # Plot progress periodically
            if plot_every and episode % plot_every == 0 and episode > 0:
                self.plot_training_progress()
            
            # Check convergence
            if len(self.running_reward) >= 100 and avg_reward > self.convergence_threshold:
                if verbose:
                    print(f"\n🎉 CONVERGENCE ACHIEVED! 🎉")
                    print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
                    print(f"Improvement over random baseline: {(avg_reward + 113.35) / 113.35 * 100:.1f}%")
                break
                
            # Adaptive learning rate decay (from paper)
            if episode == 1000:  # After marked learning
                if verbose:
                    print("📉 Reducing learning rates (as mentioned in paper)")
                self.agent.actor_optimizer.learning_rate = 0.0001
                self.agent.critic_optimizer.learning_rate = 0.001
        
        total_time = time.time() - start_time
        
        if verbose:
            print("=" * 60)
            print(f"🏁 Training completed!")
            print(f"📈 Final average reward: {avg_reward:.2f}")
            print(f"🚀 Best average reward: {self.best_avg_reward:.2f}")
            print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
            print(f"📊 Total episodes: {len(self.episode_rewards)}")
        
        return self._get_training_stats()
    
    def _get_training_stats(self):
        """Get comprehensive training statistics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'final_avg_reward': np.mean(self.running_reward) if self.running_reward else 0,
            'best_avg_reward': self.best_avg_reward,
            'convergence_achieved': self.best_avg_reward > self.convergence_threshold,
            'total_episodes': len(self.episode_rewards)
        }
    
    def plot_training_progress(self):
        """Plot training progress with multiple metrics."""
        if len(self.episode_rewards) < 10:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TD Actor-Critic Training Progress', fontsize=16)
        
        # Episode rewards
        ax1.plot(self.episode_rewards, alpha=0.6, color='blue', linewidth=0.8)
        if len(self.episode_rewards) >= 10:
            # Running average
            window = min(50, len(self.episode_rewards) // 4)
            running_avg = pd.Series(self.episode_rewards).rolling(window).mean()
            ax1.plot(running_avg, color='red', linewidth=2, label=f'Running Avg ({window})')
        ax1.axhline(y=-113.35, color='orange', linestyle='--', label='Random Baseline')
        ax1.axhline(y=self.convergence_threshold, color='green', linestyle='--', label='Target')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Actor losses
        if len(self.actor_losses) > 1:
            ax2.plot(self.actor_losses, color='purple', alpha=0.7)
            ax2.set_title('Actor Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
        
        # Critic losses
        if len(self.critic_losses) > 1:
            ax3.plot(self.critic_losses, color='orange', alpha=0.7)
            ax3.set_title('Critic Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            ax3.grid(True, alpha=0.3)
        
        # Episode lengths
        ax4.plot(self.episode_lengths, color='green', alpha=0.6)
        ax4.set_title('Episode Lengths')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_agent(self, num_episodes=10, render=False):
        """
        Evaluate the trained agent.
        
        Parameters:
        -----------
        num_episodes : int
            Number of evaluation episodes
        render : bool
            Whether to render evaluation (plot results)
            
        Returns:
        --------
        eval_stats : dict
            Evaluation statistics
        """
        print(f"🧪 Evaluating agent over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_suspension_travels = []
        eval_body_accelerations = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            suspension_data = []
            acceleration_data = []
            
            for step in range(self.max_steps_per_episode):
                # Use deterministic policy (mean action, no sampling)
                mu, std = self.agent.actor(tf.constant([[state]], dtype=tf.float32))
                action = float(mu.numpy()[0, 0])
                action = np.clip(action, -60, 60)  # Apply actuator limits
                
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                suspension_data.append(info['suspension_travel'])
                acceleration_data.append(info['body_acceleration'])
                
                state = next_state
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_suspension_travels.append(np.std(suspension_data))  # RMS-like metric
            eval_body_accelerations.append(np.std(acceleration_data))
        
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_suspension_travel_std': np.mean(eval_suspension_travels),
            'mean_body_acceleration_std': np.mean(eval_body_accelerations),
            'improvement_over_random': (np.mean(eval_rewards) + 113.35) / 113.35 * 100
        }
        
        print(f"📊 Evaluation Results:")
        print(f"   Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"   Improvement over random: {eval_stats['improvement_over_random']:.1f}%")
        print(f"   Suspension travel (std): {eval_stats['mean_suspension_travel_std']:.6f}m")
        print(f"   Body acceleration (std): {eval_stats['mean_body_acceleration_std']:.3f}m/s²")
        
        return eval_stats


# Example usage and training script
if __name__ == "__main__":
    # Import your components
    from Suspension_Model import QuarterCarModel  # Your model
    from RewardFunction import RewardFunction, SuspensionEnvironment  # Previous implementation
    from NeuralNetworkTraining import ActorCriticAgent  # Previous implementation
    
    print("🔧 Initializing training components...")
    
    # Initialize components
    car_model = QuarterCarModel()
    reward_func = RewardFunction()
    env = SuspensionEnvironment(car_model, reward_func)
    agent = ActorCriticAgent(
        actor_lr=0.001,   # ADAM paper default
        critic_lr=0.01,   # Higher for faster critic learning
        gamma=0.99        # Discount factor from paper
    )
    
    # Initialize training loop
    trainer = TrainingLoop(
        env=env, 
        agent=agent, 
        max_episodes=2000,
        max_steps_per_episode=3000  # 3 seconds at dt=0.001
    )
    
    # Start training!
    print("🎯 Starting training session...")
    training_stats = trainer.train(verbose=True, plot_every=200)
    
    # Evaluate the trained agent
    eval_stats = trainer.evaluate_agent(num_episodes=5)
    
    # Final summary
    print("\n" + "="*60)
    print("🎊 TRAINING COMPLETE! 🎊")
    print(f"🏆 Final Performance: {training_stats['final_avg_reward']:.2f}")
    print(f"🎯 Target Achieved: {'✅ YES' if training_stats['convergence_achieved'] else '❌ NO'}")
    print(f"📈 Total Episodes: {training_stats['total_episodes']}")
    print("="*60)