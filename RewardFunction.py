import numpy as np

class RewardFunction:
    """
    Reward function implementation for active suspension RL control.
    Based on the paper: "Online Reinforcement Learning-Based Control of 
    an Active Suspension System Using the Actor Critic Approach"
    """
    
    def __init__(self, k1=1000, k2=0.1):
        """
        Initialize reward function with paper's optimal parameters.
        
        Parameters:
        -----------
        k1 : float
            Weight for body velocity term (default: 1000)
        k2 : float  
            Weight for force penalty term (default: 0.1)
        """
        self.k1 = k1
        self.k2 = k2
        
    def compute_reward_v1(self, suspension_travel):
        """
        First reward function tested in paper (Equation 3).
        rt = -k * |xs - xus|
        
        Note: Paper mentions this didn't work well due to low numerical values.
        """
        return -self.k1 * abs(suspension_travel)
    
    def compute_reward_v2(self, body_velocity):
        """
        Second reward function tested in paper (Equation 4).
        rt = -k * |x_dot_s|
        
        Note: Paper mentions this couldn't eliminate steady-state error.
        """
        return -self.k1 * abs(body_velocity)
    
    def compute_reward_v3(self, body_velocity, control_force):
        """
        Third reward function - the one that worked best (Equation 5).
        rt = -k1 * (x_dot_s)^2 - k2 * |u|
        
        Parameters:
        -----------
        body_velocity : float
            Vehicle body velocity x_dot_s (m/s)
        control_force : float
            Control force u (N)
            
        Returns:
        --------
        reward : float
            Computed reward value
        """
        velocity_term = -self.k1 * (body_velocity ** 2)
        force_penalty = -self.k2 * abs(control_force)
        
        return velocity_term + force_penalty
    
    def __call__(self, body_velocity, control_force):
        """
        Default call uses the best performing reward function (v3).
        """
        return self.compute_reward_v3(body_velocity, control_force)


# Example integration with your QuarterCarModel
class SuspensionEnvironment:
    """
    RL Environment wrapper for the active suspension system.
    Integrates QuarterCarModel with reward function.
    """
    
    def __init__(self, car_model, reward_function, dt=0.001):
        """
        Initialize the RL environment.
        
        Parameters:
        -----------
        car_model : QuarterCarModel
            Your implemented quarter car model
        reward_function : RewardFunction
            Reward function instance
        dt : float
            Time step for simulation
        """
        self.car_model = car_model
        self.reward_function = reward_function
        self.dt = dt
        self.time = 0
        
        # Road profile parameters (from paper)
        self.road_amplitude = 0.02  # 0.02 m
        self.road_period = 3.0      # 3 seconds
        
    def get_road_profile(self, t):
        """
        Generate simple square wave road profile used in training.
        
        Parameters:
        -----------
        t : float
            Current time
            
        Returns:
        --------
        zr : float
            Road height
        zr_dot : float  
            Rate of road height change
        """
        # Square wave with amplitude and period from paper
        cycle_time = t % self.road_period
        if cycle_time < self.road_period / 2:
            zr = self.road_amplitude
            zr_dot = 0  # Flat sections have zero derivative
        else:
            zr = 0
            zr_dot = 0
            
        return zr, zr_dot
    
    def step(self, action):
        """
        Execute one environment step.
        
        Parameters:
        -----------
        action : float
            Control force (should be clipped to [-60, 60] N as per paper)
            
        Returns:
        --------
        state : float
            Current body velocity (used as state in paper)
        reward : float
            Computed reward (using v3 - the best performing function)
        done : bool
            Episode termination flag
        info : dict
            Additional information including all reward comparisons
        """
        # Clip action to actuator limits from paper
        action = np.clip(action, -60.0, 60.0)
        
        # Get current road profile
        zr, zr_dot = self.get_road_profile(self.time)
        
        # Update car model
        state_vector = self.car_model.update(action, zr, zr_dot, self.dt)
        
        # Extract relevant values
        xs, x_dot_s, xus, x_dot_us = state_vector
        body_velocity = x_dot_s
        
        # Get suspension travel and body acceleration for monitoring
        output = self.car_model.get_output(action)
        suspension_travel, body_acceleration = output
        
        # Compute ALL three reward functions for comparison
        reward_v1 = self.reward_function.compute_reward_v1(suspension_travel)
        reward_v2 = self.reward_function.compute_reward_v2(body_velocity)
        reward_v3 = self.reward_function.compute_reward_v3(body_velocity, action)
        
        # Use v3 as the main reward (best performing)
        reward = reward_v3
        
        # Update time
        self.time += self.dt
        
        # Simple termination condition (can be modified)
        done = False
        
        info = {
            'suspension_travel': suspension_travel,
            'body_acceleration': body_acceleration,
            'road_height': zr,
            'full_state': state_vector,
            'time': self.time,
            'reward_v1': reward_v1,
            'reward_v2': reward_v2,
            'reward_v3': reward_v3,
            'all_rewards': {
                'v1_suspension': reward_v1,
                'v2_velocity': reward_v2,
                'v3_combined': reward_v3
            }
        }
        
        return body_velocity, reward, done, info
    
    def reset(self):
        """
        Reset environment to initial state.
        
        Returns:
        --------
        state : float
            Initial body velocity (should be 0)
        """
        self.car_model.reset()
        self.time = 0
        return 0.0  # Initial body velocity
    
    def get_state(self):
        """
        Get current state (body velocity).
        
        Returns:
        --------
        state : float
            Current body velocity
        """
        return self.car_model.state[1]  # x_dot_s is at index 1


# Example usage
if __name__ == "__main__":
    # Test the reward function and environment
    from Suspension_Model import QuarterCarModel  # Your existing model
    
    # Initialize components
    car_model = QuarterCarModel()
    reward_func = RewardFunction()
    env = SuspensionEnvironment(car_model, reward_func)
    
    # Test a simple episode
    state = env.reset()
    
    # Track totals for comparison
    total_rewards = {'v1': 0, 'v2': 0, 'v3': 0}
    
    print("Testing environment - Comparing all 3 reward functions:")
    print(f"Initial state (body velocity): {state:.6f}")
    print("\nStep | Action(N) | Velocity(m/s) | Suspension(m) | Reward V1    | Reward V2    | Reward V3    | Road(m)")
    print("-" * 110)
    
    for step in range(100):
        # Random action for testing
        action = np.random.uniform(-60, 60)
        
        state, reward, done, info = env.step(action)
        
        # Accumulate totals
        total_rewards['v1'] += info['reward_v1']
        total_rewards['v2'] += info['reward_v2']
        total_rewards['v3'] += info['reward_v3']
        
        if step % 10 == 0:  # Display every 10 steps for readability
            print(f"{step:4d} | {action:8.2f} | {state:11.6f} | {info['suspension_travel']:11.6f} | "
                  f"{info['reward_v1']:10.2f} | {info['reward_v2']:10.2f} | {info['reward_v3']:10.2f} | "
                  f"{info['road_height']:7.3f}")
    
    print("-" * 110)
    print(f"\nTOTAL REWARDS COMPARISON (over 100 steps):")
    print(f"V1 (Suspension Travel): {total_rewards['v1']:10.2f}")
    print(f"V2 (Body Velocity):     {total_rewards['v2']:10.2f}")
    print(f"V3 (Combined - Best):   {total_rewards['v3']:10.2f}")
    
    print(f"\nREWARD FUNCTION ANALYSIS:")
    print(f"- V1 magnitude is typically {'high' if abs(total_rewards['v1']) > 1000 else 'low'} (paper noted low numerical values)")
    print(f"- V2 vs V3 difference: {total_rewards['v3'] - total_rewards['v2']:.2f}")
    print(f"- V3 adds force penalty to encourage efficiency")
    
    print(f"\nEnvironment setup successful! V3 is used as main reward for RL training.")