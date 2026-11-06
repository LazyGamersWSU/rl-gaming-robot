# This unit test just makes sure the new_memory function is correctly storing game play experience. 
# Checking state, action, reward, next state, and game completion are all saved as one experience.

import numpy as np
import torch
import pygame

from software.RLAgent.ale import (
    Save_Data
)

def test_memory():
    memory = Save_Data(10)

    state = np.zeros((84, 84), dtype=np.uint8)
    next_state = np.ones((84, 84), dtype=np.uint8)

    memory.new_memory(state, 1, 10.0, next_state, False)

    assert len(memory) == 1

    stored = memory.memory[0]

    assert stored[0] is state
    assert stored[1] == 1
    assert stored[2] == 10.0
    assert stored[3] is next_state
    assert stored[4] is False