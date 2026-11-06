#This unit test is making sure that in the step_environment() function the program is correctly returning the commands,
#  'no action', 'down', 'left', 'right', and 'rotate'

import numpy as np
import torch
import pygame

from software.RLAgent.ale import (
    step_environment
)

class FakeEnvironment:

    def __init__(self):
        self.actions_received = []

    def step(self, action):

        self.actions_received.append(action)

        observation = np.zeros(
            (210, 160, 3),
            dtype=np.uint8
        )

        reward = 1.0
        terminated = False
        truncated = False
        info = {}

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )


def test_step_environment_nothing():
    fake_env = FakeEnvironment()

    step_environment(fake_env, 0)

    assert fake_env.actions_received == [0]


def test_step_environment_right():
    fake_env = FakeEnvironment()

    step_environment(fake_env, 1)

    assert fake_env.actions_received == [2]


def test_step_environment_left():
    fake_env = FakeEnvironment()

    step_environment(fake_env, 2)

    assert fake_env.actions_received == [3]


def test_step_environment_down():
    fake_env = FakeEnvironment()

    step_environment(fake_env, 3)

    assert fake_env.actions_received == [4]


def test_step_environment_rotate():
    fake_env = FakeEnvironment()

    step_environment(fake_env, 4)

    assert fake_env.actions_received == [1]