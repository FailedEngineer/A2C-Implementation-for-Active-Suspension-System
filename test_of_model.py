import numpy as np
import matplotlib.pyplot as plt
from Suspension_Model import QuarterCarModel

def test_passive_response():
    """Test the passive suspension response (no active control force)."""
    model = QuarterCarModel()
    
    # Simulation parameters
    t_end = 3.0  # End time (s)
    dt = 0.001   # Time step (s)
    
    # Create time array
    t = np.arange(0, t_end, dt)
    
    # Initialize arrays to store results
    xs = np.zeros_like(t)
    xus = np.zeros_like(t)
    susp_travel = np.zeros_like(t)
    body_acc = np.zeros_like(t)
    
    # Road profile - step input (0.02m bump at t=0.5s)
    zr = np.zeros_like(t)
    zr_dot = np.zeros_like(t)
    
    # Create a step at 0.5s
    step_idx = int(0.5 / dt)
    zr[step_idx:] = 0.02
    
    # Simulation loop
    model.reset()
    for i in range(len(t)):
        # No force for passive system
        force = 0
        
        # Update the model
        state = model.update(force, zr[i], zr_dot[i], dt)
        
        # Store results
        xs[i] = state[0]        # Sprung mass position
        xus[i] = state[2]       # Unsprung mass position
        
        # Get outputs
        output = model.get_output(force)
        susp_travel[i] = output[0]  # Suspension travel
        body_acc[i] = output[1]     # Body acceleration
    
    # Plot results
    fig1 = plt.figure(num="Passive Suspension Response", figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, zr)
    plt.xlabel('Time (s)')
    plt.ylabel('Road Profile (m)')
    plt.title('Road Input')
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(t, xs)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Sprung Mass Position')
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(t, xus)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Unsprung Mass Position')
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(t, body_acc)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Body Acceleration')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('passive_response_test.png')
    # No plt.show() here to keep window open
    
    print("Passive test complete. Results have been plotted.")
    return fig1

def test_active_response():
    """Test the active suspension response with a simple PID control force."""
    model = QuarterCarModel()
    
    # Simulation parameters
    t_end = 3.0  # End time (s)
    dt = 0.001   # Time step (s)
    
    # Create time array
    t = np.arange(0, t_end, dt)
    
    # Initialize arrays to store results
    xs = np.zeros_like(t)
    xus = np.zeros_like(t)
    forces = np.zeros_like(t)
    susp_travel = np.zeros_like(t)
    body_acc = np.zeros_like(t)
    
    # Road profile - step input (0.02m bump at t=0.5s)
    zr = np.zeros_like(t)
    zr_dot = np.zeros_like(t)
    
    # Create a step at 0.5s
    step_idx = int(0.5 / dt)
    zr[step_idx:] = 0.02
    
    # Simple proportional control gains
    Kp = 500
    
    # Simulation loop
    model.reset()
    for i in range(len(t)):
        # Calculate control force using a simple P controller based on body velocity
        force = -Kp * model.state[1]  # Proportional to body velocity
        force = np.clip(force, -60, 60)  # Limit force as in the paper
        
        # Update the model
        state = model.update(force, zr[i], zr_dot[i], dt)
        
        # Store results
        xs[i] = state[0]        # Sprung mass position
        xus[i] = state[2]       # Unsprung mass position
        forces[i] = force       # Control force
        
        # Get outputs
        output = model.get_output(force)
        susp_travel[i] = output[0]  # Suspension travel
        body_acc[i] = output[1]     # Body acceleration
    
    # Plot results
    fig2 = plt.figure(num="Active Suspension Response", figsize=(12, 12))
    
    plt.subplot(5, 1, 1)
    plt.plot(t, zr)
    plt.xlabel('Time (s)')
    plt.ylabel('Road Profile (m)')
    plt.title('Road Input')
    plt.grid(True)
    
    plt.subplot(5, 1, 2)
    plt.plot(t, xs)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Sprung Mass Position')
    plt.grid(True)
    
    plt.subplot(5, 1, 3)
    plt.plot(t, xus)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Unsprung Mass Position')
    plt.grid(True)
    
    plt.subplot(5, 1, 4)
    plt.plot(t, body_acc)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Body Acceleration')
    plt.grid(True)
    
    plt.subplot(5, 1, 5)
    plt.plot(t, forces)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Control Force')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('active_response_test.png')
    # No plt.show() here to keep window open
    
    print("Active test complete. Results have been plotted.")
    return fig2

def test_square_wave_input():
    """Test the model with a square wave road profile as used in the paper."""
    model = QuarterCarModel()
    
    # Simulation parameters
    t_end = 6.0  # End time (s)
    dt = 0.001   # Time step (s)
    
    # Create time array
    t = np.arange(0, t_end, dt)
    
    # Initialize arrays to store results
    xs = np.zeros_like(t)
    xus = np.zeros_like(t)
    susp_travel = np.zeros_like(t)
    body_acc = np.zeros_like(t)
    
    # Square wave road profile as used in the paper (Fig. 6)
    # Amplitude 0.02m and period 3s
    zr = 0.02 * np.array([1 if (i % 3000 < 1500) else 0 for i in range(len(t))])
    
    # Calculate zr_dot (derivative of zr)
    zr_dot = np.zeros_like(t)
    for i in range(1, len(t)):
        zr_dot[i] = (zr[i] - zr[i-1]) / dt
    
    # Simulation loop
    model.reset()
    for i in range(len(t)):
        # No force for passive system
        force = 0
        
        # Update the model
        state = model.update(force, zr[i], zr_dot[i], dt)
        
        # Store results
        xs[i] = state[0]        # Sprung mass position
        xus[i] = state[2]       # Unsprung mass position
        
        # Get outputs
        output = model.get_output(force)
        susp_travel[i] = output[0]  # Suspension travel
        body_acc[i] = output[1]     # Body acceleration
    
    # Plot results
    fig3 = plt.figure(num="Square Wave Response", figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, zr)
    plt.xlabel('Time (s)')
    plt.ylabel('Road Profile (m)')
    plt.title('Square Wave Road Input')
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(t, xs)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Sprung Mass Position')
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(t, xus)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Unsprung Mass Position')
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(t, body_acc)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Body Acceleration')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('square_wave_test.png')
    # No plt.show() here to keep window open
    
    print("Square wave test complete. Results have been plotted.")
    return fig3

if __name__ == "__main__":
    # Run all tests
    print("Running test for passive suspension response...")
    fig1 = test_passive_response()
    
    print("\nRunning test for active suspension response...")
    fig2 = test_active_response()
    
    print("\nRunning test with square wave road profile...")
    fig3 = test_square_wave_input()
    
    # Display all figures at once
    print("\nShowing all test results. Close all windows to exit.")
    plt.show()