import numpy as np
import torch
import pygame

from software.RLAgent.ale import (
    lazygamer
)

def test_lazy_gamer_output_size():
    model = Lazy_Gamer(5)

    # Fake 84 x 84 grayscale image
    state = torch.randn(1, 1, 84, 84)

    output = model(state)

    # There should be one output for each of 5 actions
    assert output.shape == (1, 5)


def test_lazy_gamer_six_actions():
    model = Lazy_Gamer(6)

    state = torch.randn(1, 1, 84, 84)

    output = model(state)

    assert output.shape == (1, 6)