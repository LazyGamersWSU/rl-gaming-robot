#This unit test is for the parse_board() function, it turns a rendered board image into a 
#boolean array representation of the board.

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
    parse_board
)

class TestParseBoard(unittest.TestCase):
    def test_blank_frame_is_empty(self):
        #An all-gray frame parses to a completely empty board.
        self.assertFalse(parse_board(render(empty_board())).any())
 
    def test_output_shape_and_dtype(self):
        #Result should be a (22, 10) bool array.
        out = parse_board(render(empty_board()))
        self.assertEqual(out.shape, (ROWS, COLS))
        self.assertEqual(out.dtype, bool)
 
    def test_single_cell_lands_at_row_col(self):
        #A painted cell shows up at the right [row, col] (not transposed).
        out = parse_board(render(board_with((5, 3))))
        self.assertTrue(out[5, 3])
        self.assertEqual(int(out.sum()), 1)
 
    def test_roundtrip_random_boards(self):
        #render() followed by parse_board() gives back the same board.
        rng = np.random.default_rng(0)
        for _ in range(5):
            b = rng.random((ROWS, COLS)) < 0.4
            np.testing.assert_array_equal(parse_board(render(b)), b)