import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces

class CustomTetrisEnv(gym.Env):
    

    def __init__(self, render_mode=None):
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        super().__init__()
        self.rows = 10
        self.cols = 5
        
        # Actions: 0=Left, 1=Right, 2=Drop
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=np.int32)
        
        # Window & Graphics Configuration
        self.render_mode = render_mode
        self.cell_size = 80  # Size of each block square in pixels
        self.window_width = self.cols * self.cell_size
        self.window_height = self.rows * self.cell_size
        
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Grid layout state
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.block_pos = [0, self.np_random.integers(0, self.cols)]
        self.board[tuple(self.block_pos)] = 1
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.board[tuple(self.block_pos)] = 0
        
        # Move piece horizontally
        if action == 0 and self.block_pos[1] > 0:
            self.block_pos[1] -= 1
        elif action == 1 and self.block_pos[1] < self.cols - 1:
            self.block_pos[1] += 1
            
        # Standard gravity descent
        self.block_pos[0] += 1
        terminated = False
        reward = 0
        
        # Lock conditions & edge boundary hits
        if self.block_pos[0] >= self.rows or self.board[tuple(self.block_pos)] == 1:
            self.block_pos[0] -= 1
            self.board[tuple(self.block_pos)] = 1
            
            # Simple row clearing check
            for r in range(self.rows):
                if np.all(self.board[r] == 1):
                    self.board[r] = 0
                    reward += 10
            
            # Respawn new block at peak
            self.block_pos = [0, self.np_random.integers(0, self.cols)]
            if self.board[tuple(self.block_pos)] == 1:
                terminated = True
                reward -= 5
        else:
            self.board[tuple(self.block_pos)] = 1
            reward += 0.1
            
        truncated = False
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Lazy initialization of the Pygame engine window canvas
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Gymnasium Custom Tetris")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas drawing buffer
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((30, 30, 40))  # Dark Slate Blue background

        # Draw grid blocks and separation lines
        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                
                if self.board[r, c] == 1:
                    # Active structural block colored bright purple/neon magenta
                    pygame.draw.rect(canvas, (230, 50, 230), rect)
                    pygame.draw.rect(canvas, (255, 255, 255), rect, 2)  # White block border
                else:
                    # Subtle ambient structural mesh grid
                    pygame.draw.rect(canvas, (50, 50, 70), rect, 1)

        if self.render_mode == "human":
            # Push changes to display window panel
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        return self.board.copy()

    def _get_info(self):
        return {"block_position": self.block_pos}