"""
Afterstate TD learning over placements.

Each decision: list every legal placement of the falling piece, compute the
board that would result (the "afterstate"), score each with a small value
network V, and play the best one (epsilon-greedy). Learning is TD(0) on
afterstates:  V(after_t) <- r_t + gamma * max_a V(after_{t+1}).

Run:  python afterstate.py
"""
import random
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn.functional as F
from main import Display
from placement import (
    MAX_CANDS,
    N_FEATURES,
    PlacementEnv,
    StepBuffer,
    candidates,
    featurize,
)
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if present
CHECKPOINT = Path("tetris_afterstate.pt")  # where progress is saved

# --- Learning settings ---
GAMMA = 0.95            # discount: how much future value counts vs immediate reward
LEARNING_RATE = 1e-3    # size of each weight update step
BATCH_SIZE = 64         # transitions sampled per update
BUFFER_SIZE = 100_000   # replay buffer capacity
LEARNING_STARTS = 2_000 # don't learn until the buffer holds this many transitions
TARGET_SYNC_EVERY = 500 # pieces between copying the network into the target network
GRAD_STEPS_PER_PIECE = 1  # updates per piece placed

# --- Exploration: probability of a random placement, shrinking over time ---
EPSILON_START, EPSILON_END = 1.0, 0.02
EPSILON_DECAY_PIECES = 50_000  # pieces over which epsilon falls from start to end
RESUME_EPSILON = 0.10          # starting epsilon when resuming a saved run

# --- Rewards and episode limits ---
SURVIVE_REWARD = 0.1          # per piece placed
DEATH_PENALTY = -1.0          # added to the final transition when the game ends
MAX_PIECES_PER_EPISODE = 1_000
NUM_EPISODES = 100_000
RENDER = True  # draw every placement; set False for fastest training


def line_reward(cleared):
    """Reward for one placement: a survival bonus plus (lines cleared) squared,
    so clearing 4 at once (16) beats four single clears (4)."""
    return SURVIVE_REWARD + cleared**2


class ValueNet(nn.Module):
    """
    The value function V: maps an afterstate's feature vector to one number
    estimating how good that board is.

    Input: tensor, shape (batch, 16). Output: tensor, shape (batch,).
    """

    def __init__(self):
        super().__init__()
        # Sequential runs these layers in order.
        #   Linear(a, b): multiplies by a learned (a x b) weight matrix and adds a bias.
        #   ReLU: replaces negatives with 0; without it, stacked Linear layers
        #         would collapse into a single linear function.
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """Run the network. squeeze(-1) turns shape (batch, 1) into (batch,)."""
        return self.net(x).squeeze(-1)


class Learner:
    """Holds the networks and optimizer, and performs the TD updates."""

    def __init__(self):
        self.net = ValueNet().to(DEVICE)     # the network being trained
        self.target = ValueNet().to(DEVICE)  # a slowly-updated copy used to compute targets
        self.target.load_state_dict(self.net.state_dict())  # start identical
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.pieces = 0  # int: total pieces placed over all training so far

    def values(self, feats):
        """
        Score afterstates with the current network (no learning).

        Args:
            feats: float32 numpy array, shape (n, 16).
        Returns:
            float32 numpy array, shape (n,).
        """
        with torch.no_grad():  # skip gradient tracking; we're only reading values
            x = torch.as_tensor(feats, device=DEVICE)  # numpy array -> torch tensor
            return self.net(x).cpu().numpy()           # back to numpy

    def learn(self, buffer):
        """
        One gradient update on a random batch from the replay buffer.

        Returns:
            float: the loss (how far predictions were from targets).
        """
        # Sample a random batch of 64 past transitions from the replay buffer.
        # Shapes: feat (64,16), reward (64,), next_feats (64,40,16),
        #         next_mask (64,40), done (64,)
        feat, reward, next_feats, next_mask, done = buffer.sample(BATCH_SIZE)

        def t(a):
            """numpy array -> torch tensor on the right device."""
            return torch.as_tensor(a, device=DEVICE)

        # PREDICTION: what the network currently thinks each chosen afterstate
        # is worth. Shape (64,). Gradients are tracked, so this is what gets trained.
        v = self.net(t(feat))

        # TARGET: what the prediction should have been, given what happened next.
        # no_grad because the target is treated as a fixed number: we move v
        # toward it, never the other way around.
        with torch.no_grad():
            b = len(feat)  # batch size (64)

            # Score every candidate afterstate of every next position with the
            # slow-moving target network. Flatten (64,40,16) -> (2560,16) for
            # one forward pass, then reshape back to (64,40).
            nv = self.target(t(next_feats).reshape(b * MAX_CANDS, N_FEATURES))
            nv = nv.reshape(b, MAX_CANDS)

            # Ignore padding: slots where mask is False (no real candidate) get
            # -1e9 so they can never be the max. Then keep the best score per
            # transition -> shape (64,). This is "the best board I can reach next".
            best_next = nv.masked_fill(~t(next_mask), -1e9).max(dim=1).values

            # target = reward just earned + discounted value of the best next board.
            # (1 - done) is 0 when the game ended, which removes the future term
            # (and the -1e9 placeholder with it, since terminal rows have no candidates).
            target = t(reward) + GAMMA * best_next * (1.0 - t(done))

        # LOSS: how far the predictions are from their targets, averaged over the
        # batch. Smooth L1 acts like squared error for small gaps and absolute
        # error for large ones, so a single huge gap can't dominate.
        loss = F.smooth_l1_loss(v, target)

        # UPDATE: nudge the network's weights so v moves toward target.
        # Boards with good next options (low holes, room for clears) end up with high targets, boards that lead to death get low targets
        self.optimizer.zero_grad()  # clear gradients left over from the last update
        loss.backward()             # work out how each weight contributed to the loss
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)  # cap very large updates
        self.optimizer.step()       # apply the update
        return float(loss.item())   # plain Python number, for the training log

    def save(self):
        """Write networks, optimizer state and piece count to CHECKPOINT."""
        torch.save(
            {
                "net": self.net.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "pieces": self.pieces,
            },
            CHECKPOINT,
        )

    def load(self):
        """Restore from CHECKPOINT. Returns True if loaded, False if no file exists."""
        if not CHECKPOINT.exists():
            return False
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        self.net.load_state_dict(ckpt["net"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.pieces = ckpt.get("pieces", 0)
        return True


def epsilon_at(n, start):
    """
    Exploration rate after n pieces: falls in a straight line from `start`
    to EPSILON_END over EPSILON_DECAY_PIECES, then stays there.
    """
    return start + min(n / EPSILON_DECAY_PIECES, 1.0) * (EPSILON_END - start)


def train(display, env, learner):
    """
    Run the training loop until NUM_EPISODES finish or the user exits.

    Args:
        display: Display, the pygame window (from main.py).
        env: PlacementEnv, the emulator wrapper.
        learner: Learner, the networks and optimizer.
    """
    buffer = StepBuffer(BUFFER_SIZE)
    resuming = learner.pieces > 0  # bool: did we load a checkpoint?
    eps_start = RESUME_EPSILON if resuming else EPSILON_START
    start_pieces = learner.pieces  # so epsilon decays from this run's start
    recent = deque(maxlen=20)        # pieces survived in the last 20 episodes
    recent_lines = deque(maxlen=20)  # lines cleared in the last 20 episodes

    for episode in range(1, NUM_EPISODES + 1):  # one episode = one game
        env.reset(max_random_noops=60)
        # prev holds the previous placement, (afterstate features, reward).
        # It can only be stored once we see the NEXT position's candidates,
        # which is why it is stored one step late.
        prev = None
        pieces = lines = 0  # this episode's counts
        loss_sum, loss_n = 0.0, 0
        died = False

        while pieces < MAX_PIECES_PER_EPISODE:
            # 1) Read the screen and list every legal placement.
            state = env.state()
            cands = candidates(state[2], state[0]) if state is not None else []
            if not cands:  # unreadable screen or nowhere to place: game over
                died = True
                break

            # feats: float32 array, shape (n_candidates, 16), one row per candidate.
            feats = np.stack([featurize(after, c) for _, _, after, c in cands])
            # Now that this position's candidates are known, store the previous
            # transition: (prev afterstate, prev reward, these candidates, not done).
            if prev is not None:
                buffer.add(prev[0], prev[1], feats, False)

            # 2) Choose: random with probability epsilon, otherwise the candidate
            #    the network scores highest. k is an index into cands.
            epsilon = epsilon_at(learner.pieces - start_pieces, eps_start)
            if random.random() < epsilon:
                k = random.randrange(len(cands))
            else:
                k = int(np.argmax(learner.values(feats)))

            # 3) Carry it out on the emulator.
            shape, col, _, cleared = cands[k]
            _, done = env.play(shape, col)

            pieces += 1
            lines += cleared
            learner.pieces += 1
            prev = (feats[k], line_reward(cleared))

            # 4) Learn, once the buffer has enough data.
            if len(buffer) >= LEARNING_STARTS:
                for _ in range(GRAD_STEPS_PER_PIECE):
                    loss_sum += learner.learn(buffer)
                    loss_n += 1
            if learner.pieces % TARGET_SYNC_EVERY == 0:
                learner.target.load_state_dict(learner.net.state_dict())

            # 5) Draw, and check whether the user asked to exit.
            if RENDER:
                display.draw(
                    env.obs,
                    "Afterstate",
                    {"Ep": episode, "Pieces": pieces, "Lines": lines, "Eps": f"{epsilon:.2f}"},
                )
            if display.should_exit():
                learner.save()
                return
            if done:
                died = True
                break

        # Final transition: no next position, done=True, plus the death penalty.
        if died and prev is not None:
            buffer.add(prev[0], prev[1] + DEATH_PENALTY, None, True)

        recent.append(pieces)
        recent_lines.append(lines)
        avg_loss = loss_sum / loss_n if loss_n else float("nan")
        print(
            f"Ep {episode} | pieces {pieces} (avg20 {np.mean(recent):.0f}) | "
            f"lines {lines} (avg20 {np.mean(recent_lines):.1f}) | loss {avg_loss:.4f} | "
            f"eps {epsilon:.2f} | total pieces {learner.pieces}"
        )
        if episode % 20 == 0:  # periodic save
            learner.save()


def main():
    """Train directly: python afterstate.py"""
    display = Display()
    env = PlacementEnv()
    learner = Learner()
    print("Resumed from checkpoint." if learner.load() else "Starting fresh.")
    try:
        train(display, env, learner)
    except KeyboardInterrupt:  # Ctrl+C saves progress
        learner.save()
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()