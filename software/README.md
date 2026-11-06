# Tetris Afterstate Agent (software)

A reinforcement-learning agent that plays Atari Tetris (ALE) by choosing a
final placement for each falling piece. It can train in the emulator, replay a trained model, and drive a real controller
through an Arduino.

## How it works

1. A piece is falling. The agent lists every legal (orientation, column) placement.
2. For each one it simulates the resulting board (the **afterstate**) and turns it into
   16 numbers: 10 column heights plus holes, total height, bumpiness, max height,
   lines cleared and row fullness.
3. A small neural network `V` scores each afterstate. The best one is played
   (or a random one with probability epsilon while training).
4. A closed-loop executor presses rotate / left / right, checking the screen after
   each press, until the piece is where the agent wanted it.
5. Learning is TD(0) on afterstates: `V(after_t) <- r_t + gamma * max V(after_t+1)`,
   using a replay buffer and a periodically synced target network.

Rewards: `+0.1` per piece survived, `+(lines cleared)^2`, `-1` on death.

## Files

| File | Purpose |
|---|---|
| `placement.py` | Screen reading, piece shapes, placement search, features, button executor, replay buffer. |
| `afterstate.py` | Value network, learner, training loop, checkpointing. |
| `main.py` | Menu, pygame display, watch mode, Arduino controller mode. |
| `watch.py` | Watch mode: greedy play from the checkpoint, with speed control. |
| `controller.py` | Controller mode: Arduino link, USB controller reader, hardware settings. |

## Setup

```
python -m pip install -r requirements.txt
cd software
```

`pyserial` is only needed for controller mode.

## Running

```
python main.py
```

| Key | Mode | What it does |
|---|---|---|
| `1` | Train | Trains from scratch, or resumes from `tetris_afterstate.pt` |
| `2` | Watch | Greedy play from the saved checkpoint (no exploration, no learning) |
| `3` | Controller | Agent chooses pulses, Arduino presses the real buttons, USB controller input steps the emulator |
| `SPACE` | Exit | Saves the checkpoint when training |

`python afterstate.py` also trains directly.

## Checkpoint

Training saves `tetris_afterstate.pt` every 20 episodes and on exit. It holds the
network, target network, optimizer state and total pieces played, so training resumes
where it left off (with a lower starting epsilon).

This file is ignored from the repository. You will have your own local copy of the model's checkpoint, and all trainig will start from scratch.

## Key settings

In `afterstate.py`: `GAMMA`, `LEARNING_RATE`, `BATCH_SIZE`, `BUFFER_SIZE`,
`LEARNING_STARTS`, `TARGET_SYNC_EVERY`, epsilon schedule, reward constants,
`RENDER` (set `False` for fastest training).

In `main.py` (controller mode): `SERIAL_PORT`, `SERIAL_BAUD`, `ARDUINO_MAP`,
`MOVE_RELEASE_FRAMES`, `SOFT_DROP`, `DROP_RELEASE_FRAMES`, `BUTTON_ACTIONS`.

## Controller mode notes

- Every move, rotation or soft-drop row is one `"<id>\n"` serial command; the Arduino
  decides pulse width.
- `SOFT_DROP = False` means DOWN is never pressed and gravity drops the piece, which
  keeps the DOWN actuator cool.
- Pulses are logged to `input_log.json` on exit.
- Requires a trained checkpoint first.

## Test Cases
### Test 1: Simple Interactions With Game
This test is pretty straight forward, after starting the ale.py file, I manually controlled the game to make sure all inputs were being tracked. This being up, down, left, right, and both rotates. Once the game ended I went into the .json file that tracked all moves made and ctrl+f 'keyboard' to make the search easier. This showed me that all moves are being tracked properly. All moves were recorded and in the correct order, the timestamps were increased, and were accurate to the buttons pressed (no misinputs).
 
### Test 2: Controller Diagnostics Test
Added 2 simple lines of code to make sure that all the controller buttons were recognized. These lines were:
    print(f"Buttons: {controller.get_numbuttons()}")
    print(f"Axes: {controller.get_numaxes()}")
These codes correctly identified all buttons and the Up, Down, Left, Right controlls on the D-Pad (The 'Axes')
 
### Test 3: Testing all controller inputs and getting their numbers
Created a test a function (test_controller), that reads the controller inputs to make sure they are all working properly and produced numbers for each button for the arduino.
    X - 0, A - 1, B - 2, Y - 3, L - 4, R - 5, SELECT - 8, START - 9, Horizontal D-Pad direction - Axis 0, Vertical D-Pad Direction - Axis 1
 
### The Function:
    def test_controller(controller):
    print("\n--- NES CONTROLLER TEST ---")
    print("Press buttons on the controller.")
    print("Press ESC to exit.\n")
 
    running = True
 
    while running:
        for event in pygame.event.get():
 
            if event.type == pygame.QUIT:
                running = False
 
            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"BUTTON DOWN: {event.button}")
 
            elif event.type == pygame.JOYBUTTONUP:
                print(f"BUTTON UP: {event.button}")
 
            elif event.type == pygame.JOYAXISMOTION:
                print(
                    f"AXIS {event.axis}: "
                    f"{event.value:.2f}"
                )
 
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        sleep(0.01)

## Previous Versions

### Version 1

9/18/2026
Currently the RL Agent has a Deep Q-Network that stores decisions and results into files. Pytorch sends the DQN game data and gives response to actions taken by the DQN. Ale.py houses the code to call the DQN and creates a tetris enviroment. The bot can interact with the game, winning one seems to be rare thus far. The commands for manual controls are interchangable with the bot control.

Currently it does not send out USB commands, and learning is iffy to prove. Decisions are made on a frame-by-frame basis, so this may be hindering learning as it is difficult to connect individual actions like "left" to long-term rewards.
