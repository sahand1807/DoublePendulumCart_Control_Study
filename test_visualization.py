"""
Simple test script to visualize the double pendulum cart environment.
Run this to see if everything is working!
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
    """Test environment with no control (free fall)."""
    print("\n" + "=" * 60)
    print("Testing Free Fall (No Control)")
    print("=" * 60)
    
    env = DoublePendulumCartEnv()
    
    # Start with small perturbation
    initial_state = np.array([0.0, 0.2, 0.1, 0.0, 0.0, 0.0])
    obs, _ = env.reset(options={"initial_state": initial_state})
    
    print(f"\nInitial state: x={obs[0]:.3f}, θ₁={obs[1]:.3f}, θ₂={obs[2]:.3f}")
    
    states = [obs]
    rewards = []
    
    # Run for 200 steps with no control
    for i in range(200):
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
    
    # Plot positions
    axes[0].plot(time_steps, states[:, 0], label='Cart position (x)')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Free Fall Dynamics (No Control)')
    
    # Plot angles
    axes[1].plot(time_steps, states[:, 1], label='θ₁ (first pendulum)')
    axes[1].plot(time_steps, states[:, 2], label='θ₂ (second pendulum)')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
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
    print("Testing Random Control")
    print("=" * 60)
    
    env = DoublePendulumCartEnv()
    
    # Start near upright
    obs, _ = env.reset(seed=42)
    
    states = [obs]
    actions = []
    rewards = []
    
    # Run for 300 steps with random control
    for i in range(300):
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
    axes[0].set_title('Random Control Test')
    axes[0].axhline(y=2.0, color='r', linestyle='--', alpha=0.3, label='Limit')
    axes[0].axhline(y=-2.0, color='r', linestyle='--', alpha=0.3)
    axes[0].legend()
    
    # Plot angles
    axes[1].plot(time_steps, states[:, 1], label='θ₁')
    axes[1].plot(time_steps, states[:, 2], label='θ₂')
    axes[1].set_ylabel('Angles (rad)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot control action
    axes[2].plot(time_steps[:-1], actions)
    axes[2].set_ylabel('Force (N)')
    axes[2].grid(True)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
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
    print("Creating Animation")
    print("=" * 60)
    
    env = DoublePendulumCartEnv()
    
    # Collect trajectory
    initial_state = np.array([0.0, 0.3, -0.2, 0.0, 0.0, 0.0])
    obs, _ = env.reset(options={"initial_state": initial_state})
    
    states = [obs]
    
    print("\nSimulating dynamics...")
    for i in range(300):
        action = np.array([0.0])  # No control for visualization
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        
        if terminated or truncated:
            break
    
    states = np.array(states)
    print(f"Collected {len(states)} frames")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.2, 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum on Cart')
    
    # Plot elements
    cart_width = 0.3
    cart_height = 0.1
    
    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, 
                               fc='blue', alpha=0.7)
    ax.add_patch(cart_patch)
    
    line1, = ax.plot([], [], 'ro-', linewidth=3, markersize=8, label='Pendulum 1')
    line2, = ax.plot([], [], 'go-', linewidth=3, markersize=8, label='Pendulum 2')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend()
    
    def init():
        cart_patch.set_xy((0, 0))
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        return cart_patch, line1, line2, time_text
    
    def animate(frame):
        x, theta1, theta2 = states[frame, :3]
        
        # Update cart
        cart_patch.set_xy((x - cart_width/2, -cart_height/2))
        
        # First pendulum
        x1 = x + 0.5 * np.sin(theta1)
        y1 = -0.5 * np.cos(theta1)
        
        line1.set_data([x, x1], [0, y1])
        
        # Second pendulum
        x2 = x1 + 0.4 * np.sin(theta2)
        y2 = y1 - 0.4 * np.cos(theta2)
        
        line2.set_data([x1, x2], [y1, y2])
        
        time_text.set_text(f'Step: {frame}/{len(states)-1}')
        
        return cart_patch, line1, line2, time_text
    
    print("\nCreating animation (this may take a moment)...")
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(states), interval=20, blit=True)
    
    # Save animation
    print("Saving animation to results/pendulum_animation.gif...")
    anim.save('results/pendulum_animation.gif', writer='pillow', fps=30)
    print("✓ Animation saved!")
    
    plt.show()
    print("=" * 60)


def main():
    """Run all tests."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "🚀 " * 20)
    print("DOUBLE PENDULUM CART TEST SUITE")
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