import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Check TensorFlow version and import TFP accordingly
print(f"TensorFlow version: {tf.__version__}")

try:
    import tensorflow_probability as tfp
    print("TensorFlow Probability imported successfully")
except ImportError:
    print("TensorFlow Probability not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-probability==0.18.0"])
    import tensorflow_probability as tfp
    print("TensorFlow Probability installed and imported")

# Ensure GPU is available (optional check)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU detected, using CPU")

class ActorNetwork(keras.Model):
    """
    Actor Network implementation based on Table 3 (best performing model).
    Takes body velocity as input, outputs mean (μ) and std (σ) for action distribution.
    """
    
    def __init__(self, action_bound=60.0, name='actor'):
        """
        Initialize Actor Network.
        
        Parameters:
        -----------
        action_bound : float
            Maximum action value (60N from paper)
        name : str
            Network name
        """
        super(ActorNetwork, self).__init__(name=name)
        self.action_bound = action_bound
        
        # Architecture from Table 3 (best model): 5 neurons
        # Activation sequence: elu → sigmoid → ReLU → Tanh → Linear
        self.hidden1 = layers.Dense(5, activation='elu', name='actor_hidden1')
        self.hidden2 = layers.Dense(5, activation='sigmoid', name='actor_hidden2') 
        self.hidden3 = layers.Dense(5, activation='relu', name='actor_hidden3')
        self.hidden4 = layers.Dense(5, activation='tanh', name='actor_hidden4')
        
        # Output layers for mean (μ) and log_std
        self.mu_layer = layers.Dense(1, activation='linear', name='actor_mu')
        self.log_std_layer = layers.Dense(1, activation='linear', name='actor_log_std')
        
        # Initialize log_std to reasonable value (paper mentions σ = 0.5 initially)
        self.min_log_std = -20
        self.max_log_std = 2
    
    @tf.function
    def call(self, state, training=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        state : tf.Tensor
            Body velocity input (batch_size, 1)
            
        Returns:
        --------
        mu : tf.Tensor
            Mean of action distribution
        std : tf.Tensor
            Standard deviation of action distribution
        """
        # Ensure state is the right shape
        if len(tf.shape(state)) == 1:
            state = tf.expand_dims(state, -1)
        
        # Forward pass through hidden layers
        x = self.hidden1(state)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        
        # Output mean and log_std
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clip log_std to prevent numerical instability
        log_std = tf.clip_by_value(log_std, self.min_log_std, self.max_log_std)
        std = tf.exp(log_std)
        
        # Scale mean to action bounds
        mu = tf.tanh(mu) * self.action_bound
        
        return mu, std
    
    def sample_action(self, state):
        """
        Sample action from the policy distribution.
        
        Parameters:
        -----------
        state : tf.Tensor or np.array
            Body velocity
            
        Returns:
        --------
        action : tf.Tensor
            Sampled action
        log_prob : tf.Tensor
            Log probability of the action
        """
        if isinstance(state, np.ndarray):
            state = tf.constant(state, dtype=tf.float32)
        
        mu, std = self(state)
        
        # Create normal distribution
        dist = tfp.distributions.Normal(mu, std)
        
        # Sample action
        action = dist.sample()
        
        # Clip action to bounds
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        
        # Calculate log probability
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_action_prob(self, state, action):
        """
        Get log probability of a specific action.
        
        Parameters:
        -----------
        state : tf.Tensor
            Body velocity
        action : tf.Tensor
            Action taken
            
        Returns:
        --------
        log_prob : tf.Tensor
            Log probability of the action
        """
        mu, std = self(state)
        dist = tfp.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action)
        return log_prob


class CriticNetwork(keras.Model):
    """
    Critic Network implementation based on Table 3 (best performing model).
    Takes body velocity as input, outputs state value V(s).
    """
    
    def __init__(self, name='critic'):
        """
        Initialize Critic Network.
        
        Parameters:
        -----------
        name : str
            Network name
        """
        super(CriticNetwork, self).__init__(name=name)
        
        # Architecture from Table 3 (best model): 18 neurons
        # Activation sequence: ReLU → elu → Linear
        self.hidden1 = layers.Dense(18, activation='relu', name='critic_hidden1')
        self.hidden2 = layers.Dense(18, activation='elu', name='critic_hidden2')
        
        # Output layer for state value
        self.value_layer = layers.Dense(1, activation='linear', name='critic_value')
    
    @tf.function
    def call(self, state, training=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        state : tf.Tensor
            Body velocity input (batch_size, 1)
            
        Returns:
        --------
        value : tf.Tensor
            State value V(s)
        """
        # Ensure state is the right shape
        if len(tf.shape(state)) == 1:
            state = tf.expand_dims(state, -1)
        
        # Forward pass
        x = self.hidden1(state)
        x = self.hidden2(x)
        value = self.value_layer(x)
        
        return value


class ActorCriticAgent:
    """
    Complete Actor-Critic agent that combines both networks.
    """
    
    def __init__(self, 
                 actor_lr=0.001, 
                 critic_lr=0.01, 
                 gamma=0.99,
                 action_bound=60.0):
        """
        Initialize the Actor-Critic agent.
        
        Parameters:
        -----------
        actor_lr : float
            Actor learning rate (paper uses 0.001)
        critic_lr : float
            Critic learning rate (paper uses 0.01, higher than actor)
        gamma : float
            Discount factor (paper uses 0.99)
        action_bound : float
            Action space bounds (±60N)
        """
        self.gamma = gamma
        self.action_bound = action_bound
        
        # Initialize networks
        self.actor = ActorNetwork(action_bound=action_bound)
        self.critic = CriticNetwork()
        
        # Initialize optimizers (using legacy Adam for TF 2.10 compatibility)
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=critic_lr)
        
        # Build networks by calling them once
        dummy_state = tf.constant([[0.0]], dtype=tf.float32)
        self.actor(dummy_state)
        self.critic(dummy_state)
        
        print(f"Actor parameters: {self.actor.count_params()}")
        print(f"Critic parameters: {self.critic.count_params()}")
    
    def get_action(self, state):
        """
        Get action from the current policy.
        
        Parameters:
        -----------
        state : float or np.array
            Body velocity
            
        Returns:
        --------
        action : float
            Action to take
        log_prob : float
            Log probability of the action
        """
        if isinstance(state, (int, float)):
            state = tf.constant([[state]], dtype=tf.float32)
        elif isinstance(state, np.ndarray):
            if state.ndim == 0:
                state = tf.constant([[float(state)]], dtype=tf.float32)
            else:
                state = tf.constant(state.reshape(1, -1), dtype=tf.float32)
        
        action, log_prob = self.actor.sample_action(state)
        
        return float(action.numpy()[0, 0]), float(log_prob.numpy()[0, 0])
    
    def get_value(self, state):
        """
        Get state value from critic.
        
        Parameters:
        -----------
        state : float or np.array
            Body velocity
            
        Returns:
        --------
        value : float
            State value
        """
        if isinstance(state, (int, float)):
            state = tf.constant([[state]], dtype=tf.float32)
        elif isinstance(state, np.ndarray):
            if state.ndim == 0:
                state = tf.constant([[float(state)]], dtype=tf.float32)
            else:
                state = tf.constant(state.reshape(1, -1), dtype=tf.float32)
        
        value = self.critic(state)
        return float(value.numpy()[0, 0])
    
    @tf.function
    def _train_step_tf(self, state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor):
        """
        TensorFlow compiled training step for better performance.
        """
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # Get current state value and next state value
            current_value = self.critic(state_tensor)
            next_value = self.critic(next_state_tensor)
            
            # Calculate TD target and error
            td_target = tf.where(done_tensor, 
                               reward_tensor, 
                               reward_tensor + self.gamma * next_value)
            
            td_error = td_target - current_value
            
            # Critic loss (TD error squared)
            critic_loss = tf.reduce_mean(tf.square(td_error))
            
            # Actor loss (policy gradient with advantage)
            log_prob = self.actor.get_action_prob(state_tensor, action_tensor)
            actor_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(td_error))
        
        # Calculate gradients and update networks
        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        return actor_loss, critic_loss, td_error
    
    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one training step using TD advantage actor-critic.
        
        Parameters:
        -----------
        state : float
            Current state (body velocity)
        action : float
            Action taken
        reward : float
            Reward received
        next_state : float
            Next state (body velocity)
        done : bool
            Episode termination flag
            
        Returns:
        --------
        actor_loss : float
            Actor loss value
        critic_loss : float
            Critic loss value
        td_error : float
            Temporal difference error
        """
        # Convert to tensors
        state_tensor = tf.constant([[state]], dtype=tf.float32)
        action_tensor = tf.constant([[action]], dtype=tf.float32)
        reward_tensor = tf.constant([[reward]], dtype=tf.float32)
        next_state_tensor = tf.constant([[next_state]], dtype=tf.float32)
        done_tensor = tf.constant([[done]], dtype=tf.bool)
        
        actor_loss, critic_loss, td_error = self._train_step_tf(
            state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
        )
        
        return (float(actor_loss.numpy()), 
                float(critic_loss.numpy()), 
                float(td_error.numpy()[0, 0]))


# Example usage and testing
if __name__ == "__main__":
    print("Testing Actor-Critic Networks...")
    
    try:
        # Initialize agent
        agent = ActorCriticAgent()
        
        # Test with some sample states (body velocities)
        test_states = [0.0, 0.1, -0.2, 0.5, -0.3]
        
        print("\nTesting action sampling:")
        for state in test_states:
            action, log_prob = agent.get_action(state)
            value = agent.get_value(state)
            print(f"State: {state:6.2f} → Action: {action:6.2f}N, "
                  f"Log_prob: {log_prob:6.3f}, Value: {value:6.2f}")
        
        # Test training step
        print("\nTesting training step:")
        state = 0.1
        action = 10.0
        reward = -50.0
        next_state = 0.05
        done = False
        
        actor_loss, critic_loss, td_error = agent.train_step(state, action, reward, next_state, done)
        print(f"Training step completed:")
        print(f"  Actor loss: {actor_loss:.6f}")
        print(f"  Critic loss: {critic_loss:.6f}")
        print(f"  TD error: {td_error:.6f}")
        
        print("\nNetworks initialized successfully!")
        print("Ready for training loop implementation.")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        print("Please check TensorFlow and TensorFlow Probability installation:")
        print("pip install tensorflow==2.10.0")
        print("pip install tensorflow-probability==0.18.0")