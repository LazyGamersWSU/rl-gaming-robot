#This unit test is for the top_blob() function, it tests to make sure the program is always returning the topmost blob of filled cells on the board. 
#It also tests to make sure that diagonal cells are counted as connected and that a piece resting on the stack merges with it.

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
    top_blob
)

class TestTopBlob(unittest.TestCase):
    def blob_set(self, board):
        return {tuple(x) for x in top_blob(board).tolist()}
 
    def test_empty_board_returns_none(self):
        #No filled cells means no blob
        self.assertIsNone(top_blob(empty_board()))
 
    def test_lone_piece_is_four_cells(self):
        #A free-floating piece comes back as exactly its 4 cells
        cells = piece_cells("T", 3, 4)
        self.assertEqual(self.blob_set(board_with(*cells)), set(cells))
 
    def test_diagonal_cells_are_connected(self):
        #Diagonal neighbours count as one blob (8-connectivity)
        self.assertEqual(self.blob_set(board_with((3, 3), (4, 4))), {(3, 3), (4, 4)})
 
    def test_piece_touching_stack_merges_into_big_blob(self):
        #A piece resting on the stack merges with it, so the blob is > 4 cells
        stack = [(20, c) for c in range(COLS)]
        blob = top_blob(board_with(*piece_cells("T", 18, 3), *stack))
        self.assertGreater(len(blob), 4)
 
    def test_returns_topmost_blob_only(self):
        #With a piece up high and a separate stack below, only the piece is returned
        cells = piece_cells("O", 2, 0)
        board = board_with(*cells, (21, 5), (21, 6))
        self.assertEqual(self.blob_set(board), set(cells))