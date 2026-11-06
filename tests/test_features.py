#This unit test tests the features() function, making sure that that the holes in the board are counted for and the height 
# of the stack is correct.

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
    features
)

class TestFeatures(unittest.TestCase):
    def test_empty_board(self):
        #Empty board: no holes, no height, no bumpiness
        self.assertEqual(features(empty_board()), (0, 0, 0))
 
    def test_flat_floor(self):
        #A full bottom row: height 10, nothing uneven
        board = board_with(*[(21, c) for c in range(COLS)])
        self.assertEqual(features(board), (0, 10, 0))

 
    def test_hole_detection(self):
        #An empty cell under a filled cell counts as a hole
        holes, agg, bump = features(board_with((20, 0)))
        self.assertEqual((holes, agg, bump), (1, 2, 2))
 
    def test_overhang_counts_every_covered_gap(self):
        #Every empty cell below the top of a column counts, not just the first
        holes, _, _ = features(board_with((19, 2)))
        self.assertEqual(holes, 2)
