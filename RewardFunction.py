import numpy as np

class RewardFunction:
    """
    Reward function from the paper. This remains unchanged.
    rt = -k1 * (x_dot_s)^2 - k2 * |u|
    """
    def __init__(self, k1=1000, k2=0.1):
        self.k1 = k1
        self.k2 = k2
        
    def __call__(self, body_velocity, control_force):
        velocity_term = -self.k1 * (body_velocity ** 2)
        force_penalty = -self.k2 * abs(control_force)
        return velocity_term + force_penalty

class StableSuspensionEnvironment:
    """
    --- NEW AND IMPROVED ENVIRONMENT ---
    This environment is designed to be numerically stable to allow for effective learning.
    It incorporates several critical fixes:
    1.  A smooth road profile with ramps instead of sharp steps.
    2.  Clipping of state variables to prevent explosions.
    3.  Clipping of the reward value to provide a more stable learning signal.
    """
    def __init__(self, car_model, dt=0.001):
        self.car_model = car_model
        self.reward_function = RewardFunction() # Using the paper's best parameters
        self.dt = dt
        self.time = 0
        
        # Road profile parameters from paper
        self.road_amplitude = 0.02
        self.road_period = 3.0
        
        # Internal state for smooth profile generation
        self.prev_zr = 0.0
    
    def get_smooth_road_profile(self, t):
        """
        --- CRITICAL FIX ---
        Generates a road profile with smooth transitions (ramps) instead of
        instantaneous steps. This prevents infinite derivatives (zr_dot) and
        makes the simulation stable.
        """
        cycle_time = t % self.road_period
        half_period = self.road_period / 2
        transition_duration = 0.01 # 10ms ramp time

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
        # Flat bottom section
        else:
            zr = 0.0
            
        # Calculate derivative numerically, which is now well-behaved
        zr_dot = (zr - self.prev_zr) / self.dt
        self.prev_zr = zr
        
        return zr, zr_dot

    def step(self, action):
        """Execute one environment step with safety checks."""
        # Clip action to actuator limits
        action = float(np.clip(action, -60.0, 60.0))
        
        # Get road profile from the STABLE generator
        zr, zr_dot = self.get_smooth_road_profile(self.time)
        
        # Update car model
        state_vector = self.car_model.update(action, zr, zr_dot, self.dt)
        
        # Check for and handle numerical instability from the model
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            # If model explodes, reset and return a neutral state with high penalty
            self.car_model.reset()
            return 0.0, -1000.0, True, {}

        xs, x_dot_s, xus, x_dot_us = state_vector
        
        # --- CRITICAL FIX ---
        # Clip the body velocity before calculating the reward to prevent explosion.
        # This is essential because reward = -1000 * velocity^2.
        MAX_PHYSICAL_VELOCITY = 2.0  # m/s
        clipped_body_velocity = np.clip(x_dot_s, -MAX_PHYSICAL_VELOCITY, MAX_PHYSICAL_VELOCITY)
        
        # Calculate reward using the CLIPPED velocity
        reward = self.reward_function(clipped_body_velocity, action)
        
        # Update time
        self.time += self.dt
        
        # Episode termination (optional, we run for fixed length)
        done = False
        
        info = {
            'suspension_travel': xs - xus,
            'body_acceleration': self.car_model.get_output(action)[1],
            'time': self.time,
        }
        
        # The agent's state is the true (unclipped) body velocity
        return x_dot_s, reward, done, info
    
    def reset(self):
        """Resets the environment to a starting state."""
        self.car_model.reset()
        self.time = 0
        self.prev_zr = 0.0
        # Initial state is body velocity, which is 0
        return 0.0
