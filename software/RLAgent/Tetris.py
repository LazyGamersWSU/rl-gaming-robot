<<<<<<< HEAD
=======

>>>>>>> 8254410 (Chore: Updated py files for better import block organization to appease Lint)
import pygame
from tetris_env import CustomTetrisEnv

# Initialize environment with the human window render hook explicitly enabled
env = CustomTetrisEnv(render_mode="human")
obs, info = env.reset()

running = True
step = 0

print("Pop-up Window running! Click on the Pygame window to view it.")

while running:
    # 1. Listen for window closure window cross triggers
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    # 2. Select a placeholder random action choice
    action = env.action_space.sample()

    # 3. Take step inside the canvas
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if terminated or truncated:
        print(f"Episode completed at step {step}. Resetting environment...")
        obs, info = env.reset()
        step = 0

# Proper cleanup safety block
env.close()
print("Environment closed cleanly.")
