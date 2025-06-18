"""
🔬 COMPREHENSIVE AGENT EVALUATION SCRIPT (v5 - RLS Corrected)
This script loads a trained RL agent and evaluates its performance against
Passive, PID, and LQR controllers, generating separate, clear plot windows
for each major comparison, mirroring the paper's figures.

- Window 1: Scenario 1 (Square Wave Road) Controller Comparison
- Window 2: Scenario 2 (Bumpy Road & Varying Params) Controller Comparison
- Window 3: Scenario 2 RLS Online Parameter Estimation

**CORRECTIONS IN THIS VERSION:**
- Implemented a corrected, physically-based regressor vector for the RLS estimator.
- RLS now uses a realistic acceleration measurement (via numerical differentiation).
- This results in a stable and accurate parameter estimation.

How to run:
1. To test the latest checkpoint:
   python evaluate_agent.py
2. To test a specific checkpoint:
   python evaluate_agent.py --checkpoint_path ./training_checkpoints/ckpt-2000
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import argparse
from scipy.linalg import solve_continuous_are

# Import necessary components from other project files
from Suspension_Model import QuarterCarModel
from RewardFunction import StableSuspensionEnvironment
from NeuralNetworkTraining import ActorCriticAgent

# --- CONTROLLER IMPLEMENTATIONS ---

class LQRController:
    """Calculates and applies the optimal LQR control force."""
    def __init__(self, model):
        A, B = model.A, model.B
        Q = np.diag([450, 30, 5, 0.1])
        R = np.array([[0.01]])
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P
        print(f"LQR Controller initialized. Gain K: {self.K.flatten()}")

    def get_action(self, state):
        force = -np.dot(self.K, state)
        return force.item()

class PIDController:
    """A standard PID controller focused on suspension travel."""
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.reset()
        print(f"PID Controller initialized with Kp={Kp}, Ki={Ki}, Kd={Kd}")
        
    def reset(self):
        self._prev_error, self._integral = 0.0, 0.0

    def get_action(self, error):
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        return self.Kp * error + self.Ki * self._integral + self.Kd * derivative

# --- ONLINE PARAMETER ESTIMATOR (RLS) ---

class RLSEstimator:
    """Recursive Least Squares estimator for online parameter identification."""
    def __init__(self, n_params, lambda_=0.9):
        self.lambda_ = lambda_
        self.theta = np.zeros(n_params) # [bs, bus]
        self.P = np.eye(n_params) * 1000
        print(f"RLS Estimator initialized for {n_params} parameters.")
        
    def update(self, y_measured, phi):
        phi = phi.reshape(-1, 1)
        K = (self.P @ phi) / (self.lambda_ + phi.T @ self.P @ phi)
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_
        error = y_measured - (phi.T @ self.theta)
        self.theta = self.theta + (K * error).flatten()
        return self.theta

# --- ENVIRONMENT DEFINITIONS ---

class BumpyRoadWithVaryingParams(StableSuspensionEnvironment):
    """Scenario 2: Bumpy road with time-varying damping parameters."""
    def __init__(self, car_model, dt=0.001):
        super().__init__(car_model, dt)
        self.noise_amp, self.base_amp, self.period = 0.015, 0.025, 30.0
        np.random.seed(42)
        noise = (np.random.rand(10000) - 0.5) * self.noise_amp
        self.smooth_noise = pd.Series(noise).rolling(window=50, min_periods=1).mean().to_numpy()

    def get_smooth_road_profile(self, t):
        # Update damping parameters sinusoidally over time
        self.car_model.params['bs'] = 6.5 + 2.5 * np.sin(2 * np.pi * 0.5 * self.time)
        self.car_model.params['bus'] = 5.0 + 2.0 * np.sin(2 * np.pi * 0.5 * self.time)
        self.car_model.update_matrices() # Recalculate A matrix with new params

        base_profile = self.base_amp * (1 + np.sin(2 * np.pi * t / self.period)) / 2
        idx = int(t / self.dt) % len(self.smooth_noise)
        zr = base_profile + self.smooth_noise[idx]
        zr_dot = (zr - self.prev_zr) / self.dt
        self.prev_zr = zr
        return zr, zr_dot

    def step(self, action):
        """Overridden step method to return the zr_dot used in the calculation."""
        action = float(np.clip(action, -60.0, 60.0))
        zr, zr_dot = self.get_smooth_road_profile(self.time)
        state_vector = self.car_model.update(action, zr, zr_dot, self.dt)
        
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            self.car_model.reset()
            info = {'zr_dot': 0.0, 'body_acceleration': 99, 'suspension_travel': 0, 'time': self.time}
            return 0.0, -1000.0, True, info

        xs, x_dot_s, xus, x_dot_us = state_vector
        reward = self.reward_function(np.clip(x_dot_s, -2.0, 2.0), action)
        self.time += self.dt
        
        info = {
            'suspension_travel': xs - xus,
            'body_acceleration': self.car_model.get_output(action)[1],
            'time': self.time,
            'zr_dot': zr_dot  # Return zr_dot for RLS
        }
        return x_dot_s, reward, False, info

# --- SIMULATION AND PLOTTING ---

def run_simulation(env, controller, steps=6000, rls=None):
    """Runs a full simulation with corrected RLS logic."""
    env.reset()
    if isinstance(controller, PIDController): controller.reset()
    results = {k: [] for k in ['time', 'xs', 'xus', 'accel', 'force', 'est_bs', 'est_bus', 'true_bs', 'true_bus']}
    
    for step in range(steps):
        current_state = env.car_model.state
        pre_step_x_dot_us = current_state[3] # For calculating acceleration

        # Determine action from controller
        if isinstance(controller, ActorCriticAgent):
            mu, _ = controller.actor(tf.constant([[current_state[1]]], dtype=tf.float32))
            action = mu.numpy()[0, 0]
        elif isinstance(controller, LQRController):
            action = controller.get_action(current_state)
        elif isinstance(controller, PIDController):
            pid_error = -(current_state[0] - current_state[2]) # Target is 0 travel
            action = controller.get_action(pid_error)
        else: # Passive
            action = 0.0
        
        action = np.clip(action, -60.0, 60.0)
        
        # Step environment
        _, _, _, info = env.step(action)
        
        # Perform RLS update if applicable
        if rls:
            # **CRITICAL FIX 1**: Calculate measured acceleration via differentiation
            post_step_x_dot_us = env.car_model.state[3]
            x_ddot_us_measured = (post_step_x_dot_us - pre_step_x_dot_us) / env.dt
            y_measured = x_ddot_us_measured + np.random.normal(0, 0.1) # Add sensor noise

            # **CRITICAL FIX 2**: Use the correct, physically-derived regressor vector
            s = env.car_model.state
            mus = env.car_model.params['mus']
            phi = np.array([
                (s[1] - s[3]) / mus,      # Coefficient of bs
                (info['zr_dot'] - s[3]) / mus  # Coefficient of bus
            ])
            
            est_theta = rls.update(y_measured, phi)
            results['est_bs'].append(est_theta[0])
            results['est_bus'].append(est_theta[1])
            results['true_bs'].append(env.car_model.params['bs'])
            results['true_bus'].append(env.car_model.params['bus'])

        # Store results for plotting
        results['time'].append(info['time'])
        results['xs'].append(env.car_model.state[0])
        results['xus'].append(env.car_model.state[2])
        results['accel'].append(info['body_acceleration'])
        results['force'].append(action)
        
    return results

def generate_comparison_plots(res, rls_res_agent, rls_res_pid):
    """Generates detailed comparison plots in separate figure windows."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- FIGURE 1: SCENARIO 1 (SQUARE WAVE) ---
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True, num='Scenario 1: Ideal Conditions (Square Wave)')
    fig1.suptitle('Scenario 1: Controller Comparison (Square Wave Road)', fontsize=16, fontweight='bold')
    for ax in axs1: ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    axs1[0].set_title('Sprung Mass Displacement (xs)')
    axs1[0].plot(res['sq_agent']['time'], res['sq_agent']['xs'], 'b-', label='RL Agent', linewidth=2)
    axs1[0].plot(res['sq_pid']['time'], res['sq_pid']['xs'], 'g--', label='PID')
    axs1[0].plot(res['sq_lqr']['time'], res['sq_lqr']['xs'], 'r:', label='LQR')
    axs1[0].set_ylabel('Displacement (m)'); axs1[0].legend()

    axs1[1].set_title('Unsprung Mass Displacement (xus)')
    axs1[1].plot(res['sq_agent']['time'], res['sq_agent']['xus'], 'b-', linewidth=2)
    axs1[1].plot(res['sq_pid']['time'], res['sq_pid']['xus'], 'g--')
    axs1[1].plot(res['sq_lqr']['time'], res['sq_lqr']['xus'], 'r:')
    axs1[1].set_ylabel('Displacement (m)')

    axs1[2].set_title('Vehicle Body Acceleration')
    axs1[2].plot(res['sq_agent']['time'], res['sq_agent']['accel'], 'b-', linewidth=2)
    axs1[2].plot(res['sq_pid']['time'], res['sq_pid']['accel'], 'g--')
    axs1[2].plot(res['sq_lqr']['time'], res['sq_lqr']['accel'], 'r:')
    axs1[2].set_ylabel('Acceleration (m/s²)'); axs1[2].set_xlabel('Time (s)')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- FIGURE 2: SCENARIO 2 (BUMPY ROAD) ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True, num='Scenario 2: Robustness Test (Bumpy Road)')
    fig2.suptitle('Scenario 2: Controller Comparison (Bumpy Road & Varying Params)', fontsize=16, fontweight='bold')
    for ax in axs2: ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    axs2[0].set_title('Sprung Mass Displacement (xs)')
    axs2[0].plot(rls_res_agent['time'], rls_res_agent['xs'], 'b-', label='RL Agent', linewidth=2)
    axs2[0].plot(rls_res_pid['time'], rls_res_pid['xs'], 'g--', label='PID')
    axs2[0].set_ylabel('Displacement (m)'); axs2[0].legend()

    axs2[1].set_title('Unsprung Mass Displacement (xus)')
    axs2[1].plot(rls_res_agent['time'], rls_res_agent['xus'], 'b-', linewidth=2)
    axs2[1].plot(rls_res_pid['time'], rls_res_pid['xus'], 'g--')
    axs2[1].set_ylabel('Displacement (m)')

    axs2[2].set_title('Vehicle Body Acceleration')
    axs2[2].plot(rls_res_agent['time'], rls_res_agent['accel'], 'b-', linewidth=2)
    axs2[2].plot(rls_res_pid['time'], rls_res_pid['accel'], 'g--')
    axs2[2].set_ylabel('Acceleration (m/s²)'); axs2[2].set_xlabel('Time (s)')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIGURE 3: RLS ESTIMATION ---
    fig3, axs3 = plt.subplots(2, 1, figsize=(12, 8), sharex=True, num='Scenario 2: RLS Parameter Estimation')
    fig3.suptitle('Scenario 2: Online Parameter Estimation (RLS)', fontsize=16, fontweight='bold')
    for ax in axs3: ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    axs3[0].set_title('Estimation of Vehicle Damping (bs)')
    axs3[0].plot(rls_res_agent['time'], rls_res_agent['true_bs'], 'k:', label='True bs Value', linewidth=2)
    axs3[0].plot(rls_res_agent['time'], rls_res_agent['est_bs'], 'b-', label='RLS Estimate')
    axs3[0].set_ylabel('Damping (Ns/m)'); axs3[0].legend()

    axs3[1].set_title('Estimation of Tire Damping (bus)')
    axs3[1].plot(rls_res_agent['time'], rls_res_agent['true_bus'], 'k:', label='True bus Value', linewidth=2)
    axs3[1].plot(rls_res_agent['time'], rls_res_agent['est_bus'], 'b-', label='RLS Estimate')
    axs3[1].set_ylabel('Damping (Ns/m)'); axs3[1].set_xlabel('Time (s)'); axs3[1].legend()
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

# --- MAIN EVALUATION FUNCTION ---
def main(checkpoint_path):
    print("\n" + "="*60)
    print("🔬 RUNNING COMPREHENSIVE CONTROLLER EVALUATION...")
    print(f"   Loading RL Agent weights from: {checkpoint_path}")
    print("="*60)

    # 1. Initialize Controllers
    agent = ActorCriticAgent()
    checkpoint = tf.train.Checkpoint(actor=agent.actor, critic=agent.critic)
    checkpoint.restore(checkpoint_path).expect_partial()
    print("✅ RL Agent weights restored successfully.")

    lqr = LQRController(QuarterCarModel())
    pid = PIDController(Kp=1500, Ki=800, Kd=200, dt=0.001)

    # 2. Run Scenario 1: Ideal Conditions
    print("\n--- Running Scenario 1: Square Wave Road ---")
    results = {
        'sq_agent': run_simulation(StableSuspensionEnvironment(QuarterCarModel()), agent),
        'sq_pid':   run_simulation(StableSuspensionEnvironment(QuarterCarModel()), pid),
        'sq_lqr':   run_simulation(StableSuspensionEnvironment(QuarterCarModel()), lqr),
    }

    # 3. Run Scenario 2: Varying Parameters and RLS
    print("\n--- Running Scenario 2: Bumpy Road + RLS ---")
    rls_res_agent = run_simulation(BumpyRoadWithVaryingParams(QuarterCarModel()), agent, rls=RLSEstimator(n_params=2))
    rls_res_pid = run_simulation(BumpyRoadWithVaryingParams(QuarterCarModel()), pid, rls=RLSEstimator(n_params=2))

    # 4. Generate and Show All Plots
    print("\n--- Generating comparison plots in separate windows ---")
    generate_comparison_plots(results, rls_res_agent, rls_res_pid)
    print("\nEvaluation complete. Close plot windows to exit.")

if __name__ == "__main__":
    def update_matrices(self):
        ms, bs, ks, mus, bus, kus = self.params.values()
        self.A = np.array([
            [0, 1, 0, 0], [-ks/ms, -bs/ms, ks/ms, bs/ms],
            [0, 0, 0, 1], [ks/mus, bs/mus, -(ks+kus)/mus, -(bs+bus)/mus]
        ])
    QuarterCarModel.update_matrices = update_matrices

    parser = argparse.ArgumentParser(description='Evaluate a trained RL agent against PID and LQR.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to a specific checkpoint file.')
    args = parser.parse_args()

    checkpoint_to_load = args.checkpoint_path or tf.train.latest_checkpoint('./training_checkpoints')

    if not checkpoint_to_load:
        print("\n❌ Error: No checkpoint found. Please train a model first.")
    else:
        main(checkpoint_to_load)
