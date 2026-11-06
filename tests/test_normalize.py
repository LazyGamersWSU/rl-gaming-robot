#This unit test is for the normalize() function, it turns cells into a position-independent representation of the shape. 
#It is used to identify pieces and their orientations.

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
    normalize
)

class TestNormalize(unittest.TestCase):
    def test_z_piece_example_from_docstring(self):
        #Matches the docstring example: a Z piece becomes the origin-based Z shape
        self.assertEqual(
            normalize([(5, 3), (5, 4), (6, 4), (6, 5)]), ((0, 0), (0, 1), (1, 1), (1, 2))
        )
 
    def test_translation_invariant(self):
        #The same shape in two places normalizes identically
        a = normalize(piece_cells("L", 2, 1))
        b = normalize(piece_cells("L", 15, 6))
        self.assertEqual(a, b)
 
    def test_order_independent(self):
        #Input order of the cells doesn't matter
        cells = piece_cells("S", 4, 4)
        self.assertEqual(normalize(cells), normalize(cells[::-1]))
 
    def test_returns_hashable_plain_ints(self):
        #Output is a tuple of tuples of Python ints, usable as a dict key
        out = normalize(np.array(piece_cells("T", 3, 3)))
        self.assertIsInstance(out, tuple)
        self.assertTrue(all(type(v) is int for cell in out for v in cell))
        hash(out)