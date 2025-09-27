# Untitled-1.py
import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(
    duration=10.0,
    dt=0.01,
    mass=1.0,
    radius=0.1,
    viscosity=1.0,
    temperature=298.0
):
    """
    Simulates and plots the 2D Brownian motion of a particle in a fluid.

    This function solves the Langevin equation numerically to model the random
    walk of a particle subjected to random collisions from fluid molecules.

    Args:
        duration (float): Total simulation time in seconds.
        dt (float): Time step for the numerical integration in seconds.
        mass (float): Mass of the particle in kg.
        radius (float): Radius of the particle in meters.
        viscosity (float): Dynamic viscosity of the fluid in Pa*s.
        temperature (float): Temperature of the fluid in Kelvin.
    """
    # --- 1. Setup and Constants ---
    # Number of time steps
    num_steps = int(duration / dt)

    # Stokes' drag coefficient (gamma) for a spherical particle
    # γ = 6 * π * η * r
    gamma = 6 * np.pi * viscosity * radius

    # Boltzmann constant (J/K)
    k_B = 1.380649e-23

    # Diffusion coefficient from the Einstein relation
    # This links the random force magnitude to temperature and viscosity.
    D = (k_B * temperature) / gamma
    
    # The standard deviation of the random force term in the discretized equation.
    # This term scales the random numbers to have the correct physical magnitude.
    random_force_std = np.sqrt(2 * D * dt)

    # --- 2. Initialize Arrays ---
    # Initialize arrays to store position (x, y) and velocity (vx, vy)
    # We start the particle at the origin with zero initial velocity.
    pos = np.zeros((num_steps, 2))
    vel = np.zeros((num_steps, 2))

    # --- 3. Simulation Loop (Euler-Maruyama Method) ---
    # This is the core of the simulation where we step through time.
    for i in range(num_steps - 1):
        # Generate random numbers from a standard normal distribution (mean=0, std=1)
        # for each dimension (x and y). This is our N(0,1).
        random_kick = np.random.randn(2)

        # Langevin Equation (discretized):
        # v_new = v_old - (drag_force * dt) + random_force
        
        # Update velocity for both x and y dimensions
        vel[i+1] = (
            vel[i]                               # Previous velocity
            - (gamma / mass) * vel[i] * dt       # Drag term
            + (random_force_std / dt) * random_kick # Stochastic (random) term
        )
        
        # Update position using the newly calculated velocity
        # x_new = x_old + v_new * dt
        pos[i+1] = pos[i] + vel[i+1] * dt

    # --- 4. Plotting the Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the particle's path
    ax.plot(pos[:, 0], pos[:, 1], color='cornflowerblue', linewidth=1.5)

    # Mark the start and end points for clarity
    ax.plot(pos[0, 0], pos[0, 1], 'o', color='green', markersize=10, label='Start')
    ax.plot(pos[-1, 0], pos[-1, 1], 'o', color='red', markersize=10, label='End')

    ax.set_title('2D Brownian Motion of a Particle in a Fluid', fontsize=16)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.legend(fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

# --- Run the simulation with default parameters ---
if __name__ == '__main__':
    # You can change the parameters here to see how they affect the motion.
    # For example, try increasing the temperature or the viscosity.
    simulate_brownian_motion(duration=20.0, temperature=350.0)
