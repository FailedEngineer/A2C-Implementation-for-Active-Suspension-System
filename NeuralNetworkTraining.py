import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Import TensorFlow Probability
try:
    import tensorflow_probability as tfp
except ImportError:
    print("TensorFlow Probability not found. Please install: pip install tensorflow-probability")
    exit()

class ActorNetwork(keras.Model):
    """
    CUSTOM Actor Network implementation.
    This architecture matches your specific request:
    - 1 Hidden Layer (5 neurons, 'elu' activation)
    - Output for mu: Linear
    - Output for sigma: ReLU (to ensure non-negative standard deviation)
    """
    def __init__(self, action_bound=60.0, name='actor_custom'):
        super(ActorNetwork, self).__init__(name=name)
        self.action_bound = action_bound
        
        # --- CUSTOM ARCHITECTURE ---
        # 1 hidden layer with 5 neurons and 'elu' activation
        self.hidden1 = layers.Dense(5, activation='elu', name='actor_hidden1')
        
        # Output layer for mean (mu) with linear activation
        self.mu_layer = layers.Dense(1, activation='linear', name='actor_mu')
        
        # Output layer for standard deviation (sigma) with ReLU activation
        # A small epsilon is added to the output to prevent sigma from being exactly zero.
        self.std_layer = layers.Dense(1, activation='relu', name='actor_sigma')
    
    @tf.function
    def call(self, state, training=None):
        if len(tf.shape(state)) == 1:
            state = tf.expand_dims(state, -1)
        
        # Pass through the single hidden layer
        x = self.hidden1(state)
        
        # Calculate mu and std
        mu = self.mu_layer(x)
        # Add a small constant to prevent std from being zero, which would cause issues.
        std = self.std_layer(x) + 1e-6 
        
        # Scale mean to action bounds
        mu = tf.tanh(mu) * self.action_bound
        
        return mu, std
    
    # The following methods use the custom architecture from call()
    def sample_action(self, state):
        if isinstance(state, np.ndarray):
            state = tf.constant(state, dtype=tf.float32)
        
        mu, std = self(state)
        dist = tfp.distributions.Normal(mu, std)
        action = dist.sample()
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def get_action_prob(self, state, action):
        mu, std = self(state)
        dist = tfp.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action)
        return log_prob

class CriticNetwork(keras.Model):
    """
    CUSTOM Critic Network implementation.
    This architecture matches your specific request:
    - 5 Hidden Layers (5 neurons each)
    - Activations: elu -> tanh -> elu -> tanh -> elu
    - Output Layer: Linear
    """
    def __init__(self, name='critic_custom'):
        super(CriticNetwork, self).__init__(name=name)
        
        # --- CUSTOM ARCHITECTURE ---
        # 5 hidden layers with specified activations
        self.hidden1 = layers.Dense(5, activation='elu', name='critic_hidden1')
        self.hidden2 = layers.Dense(5, activation='tanh', name='critic_hidden2')
        self.hidden3 = layers.Dense(5, activation='elu', name='critic_hidden3')
        self.hidden4 = layers.Dense(5, activation='tanh', name='critic_hidden4')
        self.hidden5 = layers.Dense(5, activation='elu', name='critic_hidden5')
        
        # Output layer for state value V(s)
        self.value_layer = layers.Dense(1, activation='linear', name='critic_value')
    
    @tf.function
    def call(self, state, training=None):
        if len(tf.shape(state)) == 1:
            state = tf.expand_dims(state, -1)
        
        # Pass through all five hidden layers
        x = self.hidden1(state)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        value = self.value_layer(x)
        
        return value

class ActorCriticAgent(keras.Model):
    """
    This agent class remains the same, but it will now be initialized with
    the custom ActorNetwork and CriticNetwork defined above.
    """
    def __init__(self, actor_lr=0.001, critic_lr=0.01, gamma=0.99, action_bound=60.0):
        super(ActorCriticAgent, self).__init__()
        self.gamma = gamma
        self.action_bound = action_bound
        
        # Initialize custom networks
        self.actor = ActorNetwork(action_bound=action_bound)
        self.critic = CriticNetwork()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        # Build networks
        dummy_state = tf.constant([[0.0]], dtype=tf.float32)
        self.actor(dummy_state)
        self.critic(dummy_state)
        
        print("--- Using Custom Architectures ---")
        print(f"Actor parameters: {self.actor.count_params()}")
        print(f"Critic parameters: {self.critic.count_params()}")
        print("---------------------------------")
    
    def get_action(self, state):
        if isinstance(state, (int, float)):
            state = tf.constant([[state]], dtype=tf.float32)
        elif isinstance(state, np.ndarray) and state.ndim == 0:
            state = tf.constant([[float(state)]], dtype=tf.float32)
        
        action, log_prob = self.actor.sample_action(state)
        return float(action.numpy()[0, 0]), float(log_prob.numpy()[0, 0])
        
    @tf.function
    def train_step(self, state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor):
        """TensorFlow compiled training step for performance."""
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            current_value = self.critic(state_tensor, training=True)
            next_value = self.critic(next_state_tensor, training=True)
            
            td_target = reward_tensor + self.gamma * next_value * (1.0 - tf.cast(done_tensor, tf.float32))
            td_error = td_target - current_value
            
            critic_loss = tf.reduce_mean(tf.square(td_error))
            
            log_prob = self.actor.get_action_prob(state_tensor, action_tensor)
            actor_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(td_error))
        
        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
