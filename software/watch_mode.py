"""
Watch mode: the trained network plays Tetris in the emulator. No learning.

Started from main.py (menu option 2). Needs a saved checkpoint.
"""
import random

import numpy as np
from placement import ExitRequested, candidates, featurize

WATCH_FPS = 60  # emulator frames shown per second. 60 = real time; lower = slow motion
WATCH_RANDOM_MOVE_PROB = 0.00  # chance per piece of a random placement, so games differ (0.0 means greedy, leads to deterministic games)


def watch(display, env, learner):
    """
    Play game after game with the trained network until the user exits.

    Args:
        display: Display, the pygame window.
        env: PlacementEnv, the emulator wrapper.
        learner: Learner, with a checkpoint already loaded.
    """
    stats = {"Pieces": 0, "Lines": 0}  # dict shown in the status bar
    frames = [0]  # emulator frame counter; a list so on_frame can modify it

    def on_frame(action):
        """Called by env after every emulator frame: draw, check exit, slow down."""
        frames[0] += 1
        if frames[0] % 2 == 0:  # drawing every 2nd frame is enough and cheaper
            display.draw(env.obs, "Watch", stats)
            if display.should_exit():
                raise ExitRequested  # unwinds out of env.play()
        display.clock.tick(WATCH_FPS)  # sleep so we run at most WATCH_FPS frames/second

    env.on_frame = on_frame
    try:
        while True:  # one pass = one game
            env.reset()
            stats["Pieces"] = stats["Lines"] = 0
            while not env.done:
                state = env.state()
                cands = candidates(state[2], state[0]) if state is not None else []
                if not cands:  # unreadable screen or nowhere to place: game over
                    break

                # float32 array, shape (n_candidates, 16): one row per candidate
                feats = np.stack([featurize(after, c) for _, _, after, c in cands])
                if random.random() < WATCH_RANDOM_MOVE_PROB:
                    # Occasional random placement so games diverge from each other.
                    shape, col, _, cleared = random.choice(cands)
                else:
                    # Otherwise the network's top choice.
                    shape, col, _, cleared = cands[int(np.argmax(learner.values(feats)))]
                env.play(shape, col)
                stats["Pieces"] += 1
                stats["Lines"] += cleared
            print(f"Game over: {stats['Pieces']} pieces, {stats['Lines']} lines")
    except ExitRequested:
        pass
    finally:
        env.on_frame = None  # leave the env as we found it