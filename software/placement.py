"""
Placement-level Tetris on top of ALE/Tetris-v5.

Instead of choosing a button each frame, the agent chooses a final placement
(orientation, column) for the falling piece. A closed-loop executor presses
the buttons, checking the screen after every press, until the piece is in the
right orientation/column, then lets it fall until the next piece spawns.

Measured ALE behaviour this relies on: gravity is 1 row / 16 frames, a held
LEFT/RIGHT moves 1 column / 4 frames, a held DOWN moves 1 row / 4 frames, there
is no hard drop, and button presses are only polled every few frames, so a
single 1-frame tap can be missed (hence the closed loop).

Controller mode (main.py option 3): when `input_source` is set, the emulator is
stepped with whatever the physical controller reports over USB instead of the
action the executor asked for. The executor only decides which pulse command to
send to the Arduino; the real button press is what advances the emulator.
"""

import random
import ale_py
import gymnasium as gym
import numpy as np
from scipy.ndimage import label

gym.register_envs(ale_py)

# SCREEN: (210, 160) in pixels
# BOARD: (22, 10) in Tetris cells

COLS, ROWS = 10, 22
X0, Y0, CELL_W, CELL_H = 24, 28, 4, 8
EMPTY_GRAY = np.array([111, 111, 111], dtype=np.uint8)
_XS = X0 + CELL_W * np.arange(COLS) + CELL_W // 2
_YS = Y0 + CELL_H * np.arange(ROWS) + CELL_H // 2

# DOWN actuator heat control (used only for real hardware, see main.py).
SOFT_DROP = True          # False = never press DOWN; gravity drops the piece
DROP_RELEASE_FRAMES = 12   # (unused in this file; main.py has its own copy)

# The emulator's action numbers for Tetris. env.step() takes one of these ints.
NOOP, FIRE, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4  # FIRE = rotate


def parse_board(rgb):
    """
    Convert one emulator frame into a grid of filled/empty Tetris cells.

    Args:
        rgb: numpy.ndarray, shape (210, 160, 3), dtype uint8.
             The frame in pixels, index [y, x, color], where color is 0=red, 1=green, 2=blue

    Returns:
        numpy.ndarray, shape (22, 10), dtype bool.
        Indexed [board_row, board_col]; True means the cell is filled.
        Includes both the settled stack and the falling piece.

    How: samples one pixel at the center of each cell (_YS, _XS) and marks the cell filled if that pixel is not the empty-cell gray.
    """
    # rgb[np.ix_(_YS, _XS)] -> shape (22, 10, 3): the 3 color values per cell
    # == EMPTY_GRAY         -> shape (22, 10, 3): bool per color value
    # .all(axis=2)          -> shape (22, 10): True if all 3 values are gray
    # ~                     -> flip, so True = filled
    return ~(rgb[np.ix_(_YS, _XS)] == EMPTY_GRAY).all(axis=2)


def top_blob(full_board):
    """
    Find the group of touching filled cells that contains the topmost filled cell. Normally this is the falling piece.

    Args:
        full_board: numpy.ndarray, shape (22, 10), dtype bool.
                    The output of parse_board

    Returns:
        numpy.ndarray, shape (N, 2), dtype int; each row is (board_row, board_col) of one cell in the blob.
        N is 4 for a lone piece.
        Returns None if the board has no filled cells.

    Notes:
        Cells touching diagonally count as connected (8-connectivity).
        If the piece is touching the stack, they form one blob and N will be larger than 4. Callers check len(blob) == 4.
    """
    if not full_board.any():
        return None
    # labels: shape (22, 10), dtype int. 0 = empty, 1, 2, 3... = blob id.
    # The second return value (number of blobs) is ignored with '_'.
    labels, _ = label(full_board, structure=np.ones((3, 3)))
    # r: row index of the first row that has any filled cell.
    r = np.where(full_board.any(axis=1))[0][0]
    # c: column index of the first filled cell within that row.
    c = np.where(full_board[r])[0][0]
    # Coordinates of every cell sharing the blob id at (r, c).
    return np.argwhere(labels == labels[r, c])


def normalize(cells):
    """
    Convert cell coordinates to a position-independent shape.

    Args:
        cells: numpy.ndarray of shape (N, 2), or a list of (row, col)
               pairs, in any order.

    Returns:
        tuple of N (row, col) tuples of plain Python ints, sorted, and
        shifted so the smallest row is 0 and the smallest col is 0.
        Hashable, so it can be a dict key or set member.

    Example:
        normalize([(5, 3), (5, 4), (6, 4), (6, 5)])
        -> ((0, 0), (0, 1), (1, 1), (1, 2))   # the Z piece
    """
    cells = np.asarray(cells)
    # cells.min(axis=0): axis 0 runs down the list of cells, so this gives
    # [smallest row, smallest col]. Subtracting moves the shape to the origin.
    # .tolist() converts numpy ints to plain Python ints.
    return tuple(sorted(map(tuple, (cells - cells.min(axis=0)).tolist())))


# Each piece's starting shape: a list of 4 (row, col) cells.
# Row 0 is the top. Example, "T":   . # .
#                                   # # #
_BASE = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}


def _orientations(cells):
    """
    Return the set of distinct rotations of one piece.

    Args:
        cells: list of 4 (row, col) tuples (a piece's starting shape).

    Returns:
        set of normalized shapes (each a tuple of 4 (row, col) tuples).
        Symmetric pieces give fewer: O=1, I/S/Z=2, T/J/L=4.
    """
    arr = np.array(cells)  # shape (4, 2): one (row, col) per cell
    out = {normalize(arr)}
    for _ in range(3):
        # Rotate 90 degrees clockwise: new row = old col, new col = -old row.
        # np.stack(..., axis=1) puts the two length-4 lists side by side,
        # giving shape (4, 2) again.
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
        out.add(normalize(arr))  # normalize shifts negatives back to 0
    return out


# dict: piece letter -> set of its normalized orientations
ORIENTATIONS = {name: _orientations(c) for name, c in _BASE.items()}
# dict: normalized shape -> piece letter. Used to name the piece seen on screen.
SHAPE_TO_PIECE = {s: n for n, shapes in ORIENTATIONS.items() for s in shapes}


def drop_row(board, shape, col):
    """
    Find the row where a piece comes to rest if dropped straight down.

    Args:
        board: bool array, shape (22, 10). True = filled (settled stack only).
        shape: normalized shape, a tuple of 4 (row, col) tuples.
        col: board column for the shape's left edge.

    Returns:
        int: the board row of the shape's top edge when it lands.
        None if the piece doesn't fit at all (off the right edge, or
        overlapping filled cells at the top).
    """
    cells = np.array(shape)  # shape (4, 2)
    # rows: the 4 row offsets. cols: the 4 board columns the piece occupies.
    rows, cols = cells[:, 0], cells[:, 1] + col
    # board[rows, cols] with two arrays picks the 4 cells pairwise:
    # (rows[0], cols[0]), (rows[1], cols[1]), ...
    if cols.max() >= COLS or board[rows, cols].any():
        return None
    r = 0  # how far down the piece has moved
    while True:
        new_r = r + 1  # try one more row down
        # Stop if that would pass the floor or hit a filled cell.
        if (rows + new_r).max() >= ROWS or board[rows + new_r, cols].any():
            return r
        r = new_r


def apply_placement(board, shape, col, row):
    """
    Place a piece on a board and clear any completed lines.

    Args:
        board: bool array, shape (22, 10). Not modified.
        shape, col: as in drop_row.
        row: board row of the shape's top edge (from drop_row).

    Returns:
        (new_board, cleared): new_board is a bool array, shape (22, 10);
        cleared is the int number of lines removed.
    """
    cells = np.array(shape)
    new = board.copy()  # copy so the caller's board is untouched
    new[cells[:, 0] + row, cells[:, 1] + col] = True
    # full_rows: bool array, shape (22,). True where every cell in the row
    # is filled (.all(axis=1) checks across the columns of each row).
    full_rows = new.all(axis=1)
    cleared = int(full_rows.sum())
    if cleared:
        # Keep the unfilled rows (~full_rows) and add empty rows on top,
        # which is the same as everything above sliding down.
        new = np.vstack([np.zeros((cleared, COLS), dtype=bool), new[~full_rows]])
    return new, cleared


def candidates(board, piece_name):
    """
    List every legal placement of a piece on a board.

    Args:
        board: bool array, shape (22, 10), the settled stack.
        piece_name: "I", "O", "T", "S", "Z", "J" or "L".

    Returns:
        list of (shape, col, afterstate_board, lines_cleared) tuples:
        the orientation, its left column, the resulting board, and the
        number of lines that placement clears. Up to 40 entries.

    Note: assumes the piece can reach any column. It ignores whether it
    could actually slide there past the stack.
    """
    out = []
    for shape in ORIENTATIONS[piece_name]:
        width = max(c for _, c in shape) + 1  # columns the shape spans
        for col in range(COLS - width + 1):  # every column where it fits
            row = drop_row(board, shape, col)
            if row is None:
                continue
            after, cleared = apply_placement(board, shape, col, row)
            out.append((shape, col, after, cleared))
    return out


def features(board):
    """
    Summarize a board with three numbers that describe how good it is.

    Args:
        board: bool array, shape (22, 10). True = filled.

    Returns:
        (holes, aggregate_height, bumpiness), all int:
        - holes: empty cells that have a filled cell somewhere above them
          in the same column (hard to fill later).
        - aggregate_height: sum of the 10 column heights.
        - bumpiness: sum of height differences between neighboring columns.
    """
    # board.any(axis=0): axis 0 runs down the rows, so this checks each
    # column for any filled cell -> bool array, shape (10,).
    # board.argmax(axis=0): for each column, the row index of the first True
    # (argmax returns the first maximum, and True > False) -> int array, shape (10,).
    # An empty column would give 0 from argmax, so np.where swaps in ROWS for those.
    # first: int array, shape (10,). Row of the topmost filled cell per column.
    first = np.where(board.any(axis=0), board.argmax(axis=0), ROWS)
    # heights: int array, shape (10,). Stack height per column (0 if empty).
    heights = ROWS - first
    # Broadcasting: (22, 1) compared with (1, 10) stretches to (22, 10).
    # below[r][c] is True when row r is under the top of column c's stack.
    below = np.arange(ROWS)[:, None] > first[None, :]
    # A hole is a cell that is under the stack surface and still empty.
    holes = int((below & ~board).sum())
    # np.diff gives the 9 differences between neighboring heights.
    return holes, int(heights.sum()), int(np.abs(np.diff(heights)).sum())


class PlacementEnv:
    """
    Wraps the Tetris emulator so the agent can ask for a final placement
    instead of pressing buttons frame by frame.
    """

    def __init__(self):
        # self.env: the gymnasium emulator object.
        #   obs_type="rgb"   -> frames are (210, 160, 3) uint8 arrays
        #   frameskip=1      -> each step() is exactly one emulator frame
        #   repeat_action_probability=0.0 -> no random "sticky" repeats of
        #                       the previous button, so presses are exact
        self.env = gym.make(
            "ALE/Tetris-v5", obs_type="rgb", frameskip=1, repeat_action_probability=0.0
        )
        self.obs = None             # latest frame, ndarray (210, 160, 3); None until reset
        self.done = False           # bool: True once the game is over
        self.pending_reward = 0.0   # float: emulator score gained since play() began

        # Optional hooks, used only for hardware play (main.py mode 3).
        # Both stay None during training.
        #   on_pulse(action): called once per button press
        #   on_frame(action): called after every emulator frame
        self.on_pulse = None
        self.on_frame = None
        # Idle (NOOP) frames after each move/rotate pulse, and after each
        # soft-drop pulse. Raise them if hardware can't pulse as fast as the emulator.
        self.release = 2
        self.drop_release = 0
        # False = never press DOWN; gravity (1 row / 16 frames) drops the piece.
        self.soft_drop = SOFT_DROP

        # Controller mode only. input_source is a function returning the
        # action the physical controller reports right now. When set, it
        # overrides what the executor asks for. None = normal emulator play.
        self.input_source = None
        self.applied = NOOP        # int: action actually fed to the emulator last frame
        self.press_timeout = 45    # int: frames to wait for the controller to react
        self.missed = 0            # int: pulses the controller never reported

    # -- low level --
    def _frame(self, action):
        """Advance the emulator exactly one frame with the given action (int)."""
        if self.input_source is not None:
            action = self.input_source()  # real controller state wins
        self.applied = action
        # step() returns (frame, reward, terminated, truncated, info):
        #   terminated = game over, truncated = time limit hit.
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.pending_reward += float(reward)
        self.done = self.done or terminated or truncated
        if self.on_frame is not None:
            self.on_frame(action)

    def _follow_controller(self):
        """
        Controller mode only. Step the emulator with live controller input until
        the button has been seen pressed and then released, or we time out.
        Returns True if a press was seen, False if it was missed.
        """
        seen = False
        for _ in range(self.press_timeout):
            if self.done:
                break
            self._frame(NOOP)  # _frame replaces this with the controller's state
            if self.applied != NOOP:
                seen = True
            elif seen:
                break
        if not seen:
            self.missed += 1
        return seen

    def _press(self, action, hold=4, release=None):
        """
        One button pulse = one move, one rotation, or one soft-drop row.

        Emulator mode: hold the button `hold` frames (the game only checks
        buttons every few frames, and 4 frames is exactly one move), then
        NOOP for `release` frames. Controller mode: send the pulse command
        to the Arduino and follow the real controller instead.
        """
        release = self.release if release is None else release
        if self.on_pulse is not None and not self.done:
            self.on_pulse(action)
        if self.input_source is not None:
            self._follow_controller()
        else:
            for _ in range(hold):
                if not self.done:
                    self._frame(action)
        for _ in range(release):
            if not self.done:
                self._frame(NOOP)

    def _piece_cells(self):
        """
        Cells of the falling piece as an int array, shape (4, 2) of
        (row, col), or None if the top blob isn't exactly 4 cells
        (e.g. the piece has merged with the stack).
        """
        blob = top_blob(parse_board(self.obs))
        return blob if blob is not None and len(blob) == 4 else None

    def reset(self, max_random_noops=0, randomize=False):
        """
        Start a new game and wait for the first piece to appear. Returns state().

        Args:
            max_random_noops: int. Extra random idle frames before play starts.
            randomize: bool. If True, reseed the emulator with a fresh random
                seed so each game can have a different piece sequence. If False,
                every reset replays the same game.
        """
        if randomize:
            # A new seed makes gymnasium reload the game with a different
            # internal random state (untested for Tetris: see check below).
            self.obs, _ = self.env.reset(seed=random.randrange(2**31))
        else:
            self.obs, _ = self.env.reset()
        self.done = False
        self.pending_reward = 0.0
        for _ in range(6 + random.randint(0, max_random_noops)):
            self._frame(NOOP)
        return self.state()

    def state(self):
        """
        Read the current game situation from the screen.

        Returns:
            (piece_name, current_shape, settled_board) or None if unreadable.
            - piece_name: str, "I", "O", "T", "S", "Z", "J" or "L"
            - current_shape: the falling piece's normalized shape (tuple of 4 (row, col) tuples)
            - settled_board: bool array, shape (22, 10), the stack WITHOUT the falling piece
        """
        full = parse_board(self.obs)  # bool (22, 10): stack + falling piece
        cells = self._piece_cells()   # int (4, 2) or None
        if cells is None:
            return None
        shape = normalize(cells)
        name = SHAPE_TO_PIECE.get(shape)  # .get returns None if shape is unknown
        if name is None:
            return None
        settled = full.copy()
        settled[cells[:, 0], cells[:, 1]] = False  # erase the falling piece
        return name, shape, settled

    def play(self, target_shape, target_col, max_frames=900):
        """
        Carry out a placement: rotate, shift, then let the piece fall and lock.

        Args:
            target_shape: normalized shape the piece should end up in.
            target_col: int, board column for the shape's left edge.
            max_frames: safety limit on frames spent waiting for the piece to lock.

        Returns:
            (raw_reward, done): the emulator's own score gained (float; training
            ignores it and uses its own reward) and whether the game is over (bool).
        """
        self.pending_reward = 0.0

        # 1) Rotate until the on-screen shape matches. Closed loop: re-read the
        #    screen after every press, since a press can be missed.
        for _ in range(12):
            cells = self._piece_cells()
            if self.done or cells is None or normalize(cells) == target_shape:
                break
            self._press(FIRE)

        # 2) Shift left/right until the piece's leftmost column is the target.
        for _ in range(24):
            cells = self._piece_cells()
            if self.done or cells is None:
                break
            col = int(cells[:, 1].min())  # leftmost column of the piece right now
            if col == target_col:
                break
            self._press(RIGHT if col < target_col else LEFT)

        # 3) Let the piece fall until it locks. While falling, the piece's top
        #    row only increases. A new piece appearing makes the top row jump
        #    back up (by more than 2), which means the old piece has locked.
        cells = self._piece_cells()
        prev_top = int(cells[:, 0].min()) if cells is not None else 0
        for _ in range(max_frames // 4):  # each pass advances 4 frames
            if self.done:
                break
            if self.soft_drop:
                # One row per pulse; drop_release idle frames set the duty cycle.
                self._press(DOWN, hold=4, release=self.drop_release)
            else:
                # Gravity only: no DOWN presses at all.
                for _ in range(4):
                    if not self.done:
                        self._frame(NOOP)
            cells = self._piece_cells()
            if cells is None:
                continue  # piece merged with the stack, or a line-clear animation
            top = int(cells[:, 0].min())
            if top < prev_top - 2:
                break  # a new piece spawned, so the previous one locked
            prev_top = max(prev_top, top)

        # Let the new piece settle into view so state() can read it.
        for _ in range(2):
            if not self.done:
                self._frame(NOOP)
        return self.pending_reward, self.done

    def close(self):
        """Shut down the emulator."""
        self.env.close()


# ------------------------------------------------------------
# Features and replay storage for afterstate learning (numpy only)
# ------------------------------------------------------------

MAX_CANDS = 40  # most placements one piece can have: 4 orientations x 10 columns
N_FEATURES = COLS + 6  # length of a feature vector: 10 heights + 6 summary numbers = 16


def featurize(board, cleared):
    """
    Turn an afterstate into the fixed-size vector the neural network reads.

    Args:
        board: bool array, shape (22, 10), the board after the piece landed.
        cleared: int, lines that placement cleared.

    Returns:
        float32 array, shape (16,): the 10 column heights, then holes,
        aggregate height, bumpiness, tallest column, lines cleared, and row
        fullness. Each is divided by a rough maximum so values sit near 0-1,
        which helps the network train.
    """
    # Topmost filled row per column, then column heights (same as in features()).
    first = np.where(board.any(axis=0), board.argmax(axis=0), ROWS)
    heights = ROWS - first  # int array, shape (10,)
    holes, agg, bump = features(board)
    # board.sum(axis=1): filled cells in each row, shape (22,).
    # Dividing by COLS gives each row's fraction filled; squaring rewards
    # nearly-full rows more than half-full ones. Summed to one float.
    row_fill = float(((board.sum(axis=1) / COLS) ** 2).sum())
    return np.concatenate(
        [
            heights / ROWS,
            [holes / 20, agg / 100, bump / 50, heights.max() / ROWS, cleared / 4, row_fill / 5],
        ]
    ).astype(np.float32)


class StepBuffer:
    """
    Replay buffer: a fixed-size store of past experiences that training
    samples from at random. When full, new entries overwrite the oldest
    (a "ring buffer").

    Each entry is one transition: (afterstate features, reward, features of
    every candidate afterstate in the next position, done flag).
    """

    def __init__(self, capacity):
        """capacity: int, maximum number of transitions to keep."""
        self.capacity = capacity
        # Preallocated arrays; row j holds transition j.
        self.feat = np.zeros((capacity, N_FEATURES), np.float32)             # afterstate chosen
        self.reward = np.zeros(capacity, np.float32)                         # reward received
        self.next_feats = np.zeros((capacity, MAX_CANDS, N_FEATURES), np.float32)  # next candidates
        self.next_mask = np.zeros((capacity, MAX_CANDS), np.bool_)           # True = real candidate, False = padding
        self.done = np.zeros(capacity, np.float32)                           # 1.0 if the game ended
        self.index = 0  # int: next slot to write
        self.size = 0   # int: how many slots hold real data

    def add(self, feat, reward, next_feats, done):
        """
        Store one transition.

        Args:
            feat: float32 array, shape (16,).
            reward: float.
            next_feats: float32 array, shape (n, 16) with n <= 40, or None
                        if the game ended (no next position).
            done: bool, whether the game ended.
        """
        j = self.index
        self.feat[j] = feat
        self.reward[j] = reward
        self.done[j] = float(done)
        self.next_mask[j] = False  # clear leftovers from the entry being overwritten
        if next_feats is not None:
            n = len(next_feats)
            self.next_feats[j, :n] = next_feats  # unused slots stay as padding
            self.next_mask[j, :n] = True
        self.index = (j + 1) % self.capacity  # wrap to 0 at the end
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        """
        Pick n random transitions (with replacement).

        Returns:
            (feat (n,16), reward (n,), next_feats (n,40,16),
             next_mask (n,40), done (n,))
        """
        idx = np.random.randint(0, self.size, size=n)  # int array, shape (n,)
        return (
            self.feat[idx],
            self.reward[idx],
            self.next_feats[idx],
            self.next_mask[idx],
            self.done[idx],
        )

    def __len__(self):
        """Lets len(buffer) return the number of stored transitions."""
        return self.size


class ExitRequested(Exception):
    """Raised from an env.on_frame hook to abort env.play() when the user exits."""
    