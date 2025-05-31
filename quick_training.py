from Suspension_Model import QuarterCarModel
from NeuralNetworkTraining import ActorCriticAgent
from RewardFunction import SuspensionEnvironment, FixedRewardFunction
from TrainingLoopForNN import TrainingLoop

def quick_fix_training():
    """Quick fix to restart training with proper parameters."""
    print("🚨 EMERGENCY FIX: Restarting with proper parameters...")
    
    # 1. Fixed reward function (100x smaller penalties)
    car_model = QuarterCarModel()
    reward_func = FixedRewardFunction(k1=10, k2=0.01)
    env = SuspensionEnvironment(car_model, reward_func)
    
    # 2. Much smaller learning rates
    agent = ActorCriticAgent(
        actor_lr=0.0003,   
        critic_lr=0.003,   
        gamma=0.99
    )
    
    # 3. Shorter episodes for faster feedback
    trainer = TrainingLoop(
        env=env, 
        agent=agent, 
        max_episodes=2000,
        max_steps_per_episode=700
    )
    
    # 4. Start fresh training
    print("🎯 Target: Achieve -10 to -50 average reward")
    training_stats = trainer.train(verbose=True, plot_every=100)
    
    return training_stats

if __name__ == "__main__":
    # Run the fixed training
    training_stats = quick_fix_training()