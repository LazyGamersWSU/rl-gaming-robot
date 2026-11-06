#This unit test is making sure that the only actions being performed during runtime are only actions that are allowed and is testing the behavior when 'button mashing' happens.

import numpy as np
import torch
import pygame

from software.RLAgent.ale import (
    Lazy_Gamer,
    forward
)

def test_forward_returns_allowed_action():
    model = Lazy_Gamer(5)

    state = np.random.randint(
        0,
        256,
        size=(84, 84),
        dtype=np.uint8
    )

    allowed_actions = [0, 1, 2]

    #button_mash = 0 means the model makes the decision
    action = forward(
        model,
        state,
        allowed_actions,
        0.0
    )

    #The returned action must be one of the allowed actions
    assert action in allowed_actions


def test_forward_button_mashing_returns_allowed_action():
    model = Lazy_Gamer(5)

    state = np.random.randint(
        0,
        256,
        size=(84, 84),
        dtype=np.uint8
    )

    allowed_actions = [1, 3]

    #button_mash = 1 guarantees random selection
    for _ in range(20):

        action = forward(
            model,
            state,
            allowed_actions,
            1.0
        )

        assert action in allowed_actions