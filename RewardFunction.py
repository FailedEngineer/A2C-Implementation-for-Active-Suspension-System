import numpy as np

class RewardFunction:
    """
    Reward function from the paper.
    rt = -k1 * (x_dot_s)^2 - k2 * |u|
    """
    def __init__(self, k1=1000, k2=0.1):
        self.k1 = k1
        self.k2 = k2
        
    def __call__(self, body_velocity, control_force):
        """Calculates the reward based on the raw (unclipped) body velocity."""
        velocity_term = -self.k1 * (body_velocity ** 2)
        force_penalty = -self.k2 * abs(control_force)
        return velocity_term + force_penalty

class StableSuspensionEnvironment:
    """
    The training environment for the RL agent.

    **MODIFICATIONS FOR PAPER ACCURACY:**
    1.  Uses a sharp square wave road profile with instantaneous transitions.
    2.  Calculates reward based on the raw, unclipped body velocity.
    """
    def __init__(self, car_model, dt=0.001):
        self.car_model = car_model
        # Use the unclipped reward function
        self.reward_function = RewardFunction(k1=1000, k2=0.1)
        self.dt = dt
        self.time = 0
        
        # Road profile parameters from paper's training scenario
        self.road_amplitude = 0.02
        self.road_period = 3.0
        
        # Internal state for road profile generation
        self.prev_zr = 0.0
    
    def get_road_profile(self, t):
        """
        Generates a perfect square wave road profile with sharp transitions,
        as described for the training scenario in the paper. The derivative (zr_dot)
        will be a large spike at the transitions.
        """
        cycle_time = t % self.road_period
        half_period = self.road_period / 2
        
        # Instantaneous step up/down
        zr = self.road_amplitude if cycle_time < half_period else 0.0
            
        # Calculate derivative numerically
        zr_dot = (zr - self.prev_zr) / self.dt
        self.prev_zr = zr
        
        return zr, zr_dot

    def step(self, action):
        """Execute one environment step."""
        # Clip action to actuator limits
        action = float(np.clip(action, -60.0, 60.0))
        
        # Get the sharp road profile
        zr, zr_dot = self.get_road_profile(self.time)
        
        # Update car model
        state_vector = self.car_model.update(action, zr, zr_dot, self.dt)
        
        # Handle numerical instability from the model
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            self.car_model.reset()
            # Return a large penalty if the model becomes unstable
            return 0.0, -10000.0, True, {}

        xs, x_dot_s, xus, x_dot_us = state_vector
        
        # --- MODIFICATION ---
        # Calculate reward using the RAW (unclipped) body velocity
        reward = self.reward_function(x_dot_s, action)
        
        # Update time
        self.time += self.dt
        
        # Episode termination (we run for fixed length in training)
        done = False
        
        info = {
            'suspension_travel': xs - xus,
            'body_acceleration': self.car_model.get_output(action)[1],
            'time': self.time,
        }
        
        # The agent's state is the body velocity
        return x_dot_s, reward, done, info
    
    def reset(self):
        """Resets the environment to a starting state."""
        self.car_model.reset()
        self.time = 0
        self.prev_zr = 0.0
        # Initial state is body velocity, which is 0
        return 0.0
