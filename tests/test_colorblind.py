#This unit test is verifying that the colorblind fuction is converting the game image into the expected size, the expected type, 
# and the correct coloring

import numpy as np
import torch
import pygame

from software.RLAgent.ale import (
    colorblind
)

def test_colorblind_output_shape():
    #Creates a fake RGB Tetris screenshot
    observation = np.random.randint(
        0,
        256,
        size=(210, 160, 3),
        dtype=np.uint8
    )

    result = colorblind(observation)

    assert result.shape == (84, 84)


def test_colorblind_output_type():
    observation = np.random.randint(
        0,
        256,
        size=(210, 160, 3),
        dtype=np.uint8
    )

    result = colorblind(observation)

    # The function should return uint8 data
    assert result.dtype == np.uint8


def test_colorblind_is_grayscale():
    observation = np.random.randint(
        0,
        256,
        size=(210, 160, 3),
        dtype=np.uint8
    )

    result = colorblind(observation)

    #A grayscale image should have only height and width
    assert result.ndim == 2