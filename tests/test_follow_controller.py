import controller_mode 
import main
import placement 
import watch_mode
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from placement import (
    follow_controller
)

class TestFollowController(unittest.TestCase):
    def setUp(self):
        self.env, self.fake = make_env()
        self.env.reset()
        self.fake.actions.clear()
 
    def test_press_then_release_returns_true(self):
        #Stops as soon as a press has been seen and released.
        script = iter([NOOP, RIGHT, RIGHT, NOOP, NOOP, NOOP])
        self.env.input_source = script.__next__
        self.assertTrue(self.env._follow_controller())
        self.assertEqual(self.fake.actions, [NOOP, RIGHT, RIGHT, NOOP])
        self.assertEqual(self.env.missed, 0)
 
    def test_no_press_times_out_and_counts_a_miss(self):
        #If the controller never reports a press, we wait press_timeout frames and log a miss
        self.env.press_timeout = 5
        self.env.input_source = lambda: NOOP
        self.assertFalse(self.env._follow_controller())
        self.assertEqual(len(self.fake.actions), 5)
        self.assertEqual(self.env.missed, 1)
 
    def test_held_button_counts_as_seen(self):
        #A button still held at timeout still counts as a seen press (no miss)
        self.env.press_timeout = 4
        self.env.input_source = lambda: LEFT
        self.assertTrue(self.env._follow_controller())
        self.assertEqual(self.env.missed, 0)
 
    def test_stops_when_game_ends(self):
        #No more frames are stepped once the game is over.
        self.env.done = True
        self.env.input_source = lambda: NOOP
        self.env._follow_controller()
        self.assertEqual(self.fake.actions, [])