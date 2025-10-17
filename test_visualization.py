"""
Simple test script to visualize the double pendulum cart environment.
Run this to see if everything is working!

ANGLE CONVENTION:
- θ₁, θ₂ are absolute angles from vertical (downward)
- θ = 0: hanging down
- θ = π: upright (inverted)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# Add env to path
sys.path.insert(0, os.path.dirname(__file__))

from env.double_pendulum_cart_env import DoublePendulumCartEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing Double Pendulum Cart Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = DoublePendulumCartEnv()
    print("   ✓ Environment created successfully")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Initial observation: {obs}")
    print(f"   ✓ Angles: θ₁={np.degrees(obs[1]):.1f}°, θ₂={np.degrees(obs[2]):.1f}°")
    print(f"   ✓ Initial info: {info}")
    
    # Take a few steps
    print("\n3. Taking 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: reward={reward:.3f}, terminated={terminated}")
        
        if terminated or truncated:
            print("   Episode ended early")
            break
    
    print("\n✓ Basic functionality test passed!")
    print("=" * 60)


def test_with_zero_control():
    """Test environment with no control (free fall from upright)."""
    print("\n" + "=" * 60)
    print("Testing Free Fall from Upright (No Control)")
    print("=" * 60)
    
    # Create environment WITHOUT termination on fall (for visualization)
    env = DoublePendulumCartEnv(terminate_on_fall=False)
    
    # Start near upright: θ₁ ≈ π, θ₂ ≈ π
    initial_state = np.array([0.0, np.pi + 0.2, np.pi + 0.1, 0.0, 0.0, 0.0])
    obs, _ = env.reset(options={"initial_state": initial_state})
    
    print(f"\nInitial state (absolute angles):")
    print(f"  x={obs[0]:.3f}, θ₁={obs[1]:.3f} rad ({np.degrees(obs[1]):.1f}°)")
    print(f"  θ₂={obs[2]:.3f} rad ({np.degrees(obs[2]):.1f}°)")
    print(f"  Upright is at π = {np.pi:.3f} rad (180°)")
    
    states = [obs]
    rewards = []
    
    # Run for 500 steps with no control (longer to see full dynamics)
    for i in range(500):
        action = np.array([0.0])  # No control
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        rewards.append(reward)
        
        if terminated or truncated:
            print(f"\nEpisode terminated at step {i+1}")
            break
    
    states = np.array(states)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    time_steps = np.arange(len(states))
    
    # Plot cart position
    axes[0].plot(time_steps, states[:, 0], label='Cart position (x)')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Free Fall Dynamics from Upright (No Control)')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot angles (absolute convention)
    axes[1].plot(time_steps, states[:, 1], label='θ₁ (first pendulum)', color='red')
    axes[1].plot(time_steps, states[:, 2], label='θ₂ (second pendulum)', color='green')
    axes[1].axhline(y=np.pi, color='b', linestyle='--', alpha=0.5, label='Upright (π)')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.3, label='Hanging (0)')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(-0.5, 2*np.pi + 0.5)
    
    # Plot rewards
    axes[2].plot(time_steps[:-1], rewards)
    axes[2].set_ylabel('Reward')
    axes[2].set_xlabel('Time Step')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/free_fall_test.png', dpi=150)
    print(f"\n✓ Plot saved to results/free_fall_test.png")
    plt.show()
    
    print("=" * 60)


def test_with_random_control():
    """Test environment with random control."""
    print("\n" + "=" * 60)
    print("Testing Random Control (from upright)")
    print("=" * 60)
    
    env = DoublePendulumCartEnv()
    
    # Start near upright
    obs, _ = env.reset(seed=42)
    
    states = [obs]
    actions = []
    rewards = []
    
    # Run for 500 steps with random control (longer episode)
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        states.append(obs)
        actions.append(action[0])
        rewards.append(reward)
        
        if terminated or truncated:
            print(f"\nEpisode terminated at step {i+1}")
            break
    
    states = np.array(states)
    actions = np.array(actions)
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    
    time_steps = np.arange(len(states))
    
    # Plot cart position
    axes[0].plot(time_steps, states[:, 0])
    axes[0].set_ylabel('Cart x (m)')
    axes[0].grid(True)
    axes[0].set_title('Random Control Test (starting near upright)')
    axes[0].axhline(y=2.0, color='r', linestyle='--', alpha=0.3, label='Limit')
    axes[0].axhline(y=-2.0, color='r', linestyle='--', alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].legend()
    
    # Plot angles
    axes[1].plot(time_steps, states[:, 1], label='θ₁', color='red')
    axes[1].plot(time_steps, states[:, 2], label='θ₂', color='green')
    axes[1].axhline(y=np.pi, color='b', linestyle='--', alpha=0.5, label='Upright (π)')
    axes[1].set_ylabel('Angles (rad)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot control action
    axes[2].plot(time_steps[:-1], actions * 20.0)  # Scale to actual force
    axes[2].set_ylabel('Force (N)')
    axes[2].grid(True)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].axhline(y=20, color='r', linestyle='--', alpha=0.3, label='Max')
    axes[2].axhline(y=-20, color='r', linestyle='--', alpha=0.3)
    axes[2].legend()
    
    # Plot reward
    axes[3].plot(time_steps[:-1], rewards)
    axes[3].set_ylabel('Reward')
    axes[3].set_xlabel('Time Step')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/random_control_test.png', dpi=150)
    print(f"\n✓ Plot saved to results/random_control_test.png")
    plt.show()
    
    print("=" * 60)


def animate_pendulum():
    """Create an animation of the pendulum."""
    print("\n" + "=" * 60)
    print("Creating Animation (Free Fall from Upright)")
    print("=" * 60)
    
    env = DoublePendulumCartEnv()
    
    # Start near upright
    initial_state = np.array([0.0, np.pi + 0.3, np.pi - 0.2, 0.0, 0.0, 0.0])
    obs, _ = env.reset(options={"initial_state": initial_state})
    
    states = [obs]
    
    print("\nSimulating dynamics...")
    for i in range(100):  # Longer simulation
        action = np.array([0.0])  # No control for visualization
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        
        # Progress indicator
        if i % 10 == 0:
            print(f"  Frame {i}/100: θ₁={np.degrees(obs[1]):.1f}°, θ₂={np.degrees(obs[2]):.1f}°")
        
        if terminated or truncated:
            print(f"  Episode ended at step {i}")
            break
    
    states = np.array(states)
    print(f"Collected {len(states)} frames")
    
    # Create animation with larger figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2.0, 2.0)  # Wider view
    ax.set_ylim(-1.5, 1.5)  # Taller view
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Double Pendulum on Cart - Free Fall from Upright', fontsize=14, fontweight='bold')
    
    # Plot elements with better visibility
    cart_width = 0.4
    cart_height = 0.15
    
    # Draw rail
    ax.plot([-2, 2], [0, 0], 'k-', linewidth=2, alpha=0.3, label='Rail')
    
    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, 
                               fc='steelblue', ec='black', linewidth=2, alpha=0.8)
    ax.add_patch(cart_patch)
    
    line1, = ax.plot([], [], 'o-', color='red', linewidth=4, markersize=10, 
                     markerfacecolor='darkred', label='Link 1')
    line2, = ax.plot([], [], 'o-', color='green', linewidth=4, markersize=10,
                     markerfacecolor='darkgreen', label='Link 2')
    tip_marker, = ax.plot([], [], 'o', color='gold', markersize=15, 
                         markeredgecolor='orange', markeredgewidth=2)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    ax.legend(loc='upper right', fontsize=10)
    
    def init():
        cart_patch.set_xy((0, 0))
        line1.set_data([], [])
        line2.set_data([], [])
        tip_marker.set_data([], [])
        time_text.set_text('')
        return cart_patch, line1, line2, tip_marker, time_text
    
    def animate(frame):
        x, theta1, theta2 = states[frame, :3]
        
        # Update cart
        cart_patch.set_xy((x - cart_width/2, -cart_height/2))
        
        # First pendulum (absolute angle from vertical)
        x1 = x + 0.5 * np.sin(theta1)
        y1 = -0.5 * np.cos(theta1)
        
        line1.set_data([x, x1], [0, y1])
        
        # Second pendulum (also absolute angle from vertical)
        x2 = x + 0.5 * np.sin(theta1) + 0.4 * np.sin(theta2)
        y2 = -0.5 * np.cos(theta1) - 0.4 * np.cos(theta2)
        
        line2.set_data([x1, x2], [y1, y2])
        tip_marker.set_data([x2], [y2])
        
        # Calculate time and energy
        t = frame * 0.01 * 5  # timestep * frame_skip
        
        time_text.set_text(
            f'Time: {t:.2f}s | Frame: {frame}/{len(states)-1}\n'
            f'Cart: x={x:.3f}m\n'
            f'Link 1: θ₁={np.degrees(theta1):.1f}° ({theta1:.3f} rad)\n'
            f'Link 2: θ₂={np.degrees(theta2):.1f}° ({theta2:.3f} rad)\n'
            f'Upright: 180° (π rad)'
        )
        
        return cart_patch, line1, line2, tip_marker, time_text
    
    print("\nCreating animation (this may take a moment)...")
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(states), interval=40, blit=True)  # Slower: 40ms per frame
    
    # Save animation
    print("Saving animation to results/pendulum_animation.gif...")
    anim.save('results/pendulum_animation.gif', writer='pillow', fps=25)  # 25 fps (slower)
    print("✓ Animation saved!")
    
    plt.show()
    print("=" * 60)


def main():
    """Run all tests."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "🚀 " * 20)
    print("DOUBLE PENDULUM CART TEST SUITE")
    print("Angle Convention: θ=0 (down), θ=π (upright)")
    print("🚀 " * 20)
    
    try:
        # Test 1: Basic functionality
        test_basic_functionality()
        
        # Test 2: Free fall
        test_with_zero_control()
        
        # Test 3: Random control
        test_with_random_control()
        
        # Test 4: Animation
        print("\n" + "=" * 60)
        response = input("\nDo you want to create an animation? (y/n): ")
        if response.lower() == 'y':
            animate_pendulum()
        
        print("\n" + "✅ " * 20)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ " * 20)
        print("\nCheck the 'results' folder for plots and animations.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()