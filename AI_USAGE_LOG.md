# AI Usage & Verification Log

**Project Name:** RL Gaming Robot

---

## Overview & AI Usage Statement

### Overview & AI Usage Statement

AI tools were used throughout the project as a supplemental development resource for brainstorming, troubleshooting, and improving implementation approaches. AI-generated code and recommendations were reviewed, tested, and modified by the project team before being incorporated into the project. The team remains responsible for understanding and verifying all implemented solutions. AI was not treated as a replacement for independent research, engineering judgment, or testing.

### Responsible AI Usage

AI was used responsibly by treating its output as a starting point rather than an authoritative source. Generated code was reviewed for correctness, tested in the project environment, and modified when necessary. Project decisions and final implementations were verified by the team, and AI assistance was documented in this log to maintain transparency and accountability.

---

## Entry 1: Prototype 1 - SNES USB Controller Input Mapping Experiment

* **Developer:** Sam Culp
* **Date:** August 31, 2026  
* **Tool Used:** ChatGPT  
* **Associated Git Issue:** Test SNES USB controller input mapping  
* **Associated Feature Branch:** `controller-input-test`  

### Exact Prompt Submitted:

> "quick python script for testing input on this snes usb controller i just got from walmart"

### AI Output Summary & Code Generated

AI generated a Python script using the `pygame` library to:

* Detect connected USB controllers.
* Display the controller name.
* Display the number of buttons, axes, and hats detected.
* Print button press events.
* Print button release events.
* Print joystick axis movement values.
* Print D-pad or hat movement values when applicable.

The generated experiment allowed the physical SNES USB controller to be tested interactively.

### Human Testing, Verification & Results

The controller was manually tested using the generated Python experiment. The following input mappings were observed and recorded:

#### Button Mapping

| SNES Button | Input Mapping |
|---|---:|
| X | Button 0 |
| A | Button 1 |
| B | Button 2 |
| Y | Button 3 |
| Select | Button 8 |
| Start | Button 9 |

#### D-Pad Mapping

The controller D-pad was detected as two joystick axes rather than individual buttons or a joystick hat.

| Direction | Axis | Value When Pressed |
|---|---:|---:|
| Left | Axis 0 | -1 |
| Right | Axis 0 | 1 |
| Up | Axis 1 | -1 |
| Down | Axis 1 | 1 |
| Released | Axis 0 / Axis 1 | Approximately -0.01 |

### Human Review & Modifications Identified

Testing identified that the D-pad does not return exactly `0` when released. Instead, the controller reports a value of approximately `-0.01`.

Based on this result, future controller input handling should implement a dead zone rather than checking for an exact axis value of zero.

Example dead zone approach:

```python
DEADZONE = 0.5

if axis_0 < -DEADZONE:
    print("LEFT")
elif axis_0 > DEADZONE:
    print("RIGHT")

if axis_1 < -DEADZONE:
    print("UP")
elif axis_1 > DEADZONE:
    print("DOWN")
```

---

## Entry 2: CI/CD Infrastructure - Pytest Unit Test Setup

* **Developer:** Sam Culp
* **Date:** September 8, 2026  
* **Tool Used:** ChatGPT  
* **Associated Git Issue:** CI/CD Infrastructure & Test Logs  
* **Associated Feature Branch:** `chore/cicd-implementation`  

### Exact Prompts Submitted:

> "whats best practice for virtual environments when i have a bunch of python tests like this"

> "Record pygame as a project dependency"

> "what directory should tests go under"

> "next"

> "okay next"

> "run tests failed. said no module named pytest but install dependecies worked and pytest was included in that"

> "add this conversation about setting up pytest to this log"

### AI Output Summary & Code Generated

ChatGPT provided guidance for establishing a Python unit-testing structure suitable for CI/CD. The guidance included:

* Moving the Python virtual environment from the individual hardware experiment directory to the repository root.
* Using one project-level virtual environment rather than separate virtual environments for individual test files.
* Adding `pygame` and `pytest` as project dependencies in `requirements.txt`.
* Keeping the virtual environment out of source control through `.gitignore`.
* Creating a top-level `tests/` directory for automated unit tests.
* Renaming the hardware controller experiment so pytest would not automatically discover it as a test module.
* Separating executable controller code from importable code by placing the controller program inside a `main()` function and using an `if __name__ == "__main__":` guard.
* Creating a unit-testable `get_button_action()` function and a corresponding pytest test.
* Configuring GitHub Actions to install dependencies and execute tests on pull requests.

### Human Testing, Verification & Results

The initial pytest execution failed because the hardware controller experiment was automatically collected as a test module and attempted to access a physical controller.

After restructuring the controller experiment, pytest was able to collect and execute the unit test without requiring physical hardware.

The final local test execution successfully reported:

```text
1 passed
```

The GitHub Actions workflow was then configured to install dependencies from `requirements.txt` and execute:

```text
python -m pytest
```

An initial CI failure was traced to an empty `requirements.txt`. After adding the required dependencies:

```text
pygame
pytest
```

the GitHub Actions test job successfully passed.

### Human Review & Modifications Identified

Human review determined that the original controller experiment was an interactive hardware test rather than a unit-testable module. The code was therefore modified so importing the module does not immediately initialize the controller or terminate when no controller is connected.

The CI workflow was also verified through a pull request, confirming that the automated pytest execution can run successfully in GitHub Actions without physical controller hardware.

---

## Entry 3: CI/CD Infrastructure - Ruff Linting

* **Developer:** Sam Culp
* **Date:** September 9, 2026  
* **Tool Used:** ChatGPT  
* **Associated Git Issue:** CI/CD Infrastructure & Test Logs  
* **Associated Feature Branch:** `chore/cicd-implementation`

### Exact Prompts Submitted:

> "lets continue with linting"

> "all checks passed"

> "it needs to be included in the yaml right?"

### AI Output Summary & Code Generated

ChatGPT provided guidance for adding Ruff as the project's Python linting tool. The guidance included:

* Adding `ruff` to `requirements.txt` alongside `pygame` and `pytest`.
* Installing Ruff in the project virtual environment.
* Running `python -m ruff check .` locally to identify linting issues.
* Reviewing Ruff's reported issues rather than automatically applying fixes.
* Adding a GitHub Actions workflow step to execute `python -m ruff check .` so linting is automatically performed in CI on pull requests.

### Human Testing, Verification & Results

The initial local Ruff check identified four issues: two import-formatting issues and two uses of `quit()` where Ruff recommended `sys.exit()`.

The controller input module was modified to import `sys` and replace both `quit()` calls with `sys.exit()`.

Ruff was run again with:

```text
python -m ruff check .
```

---

## Entry 4: ALE Modification

* **Developer:** Trystan Donmoyer
* **Date:** September 15, 2026  
* **Tool Used:** VS Copilot  
* **Associated Git Issue:** Updating ALE.py file  
* **Associated Feature Branch:** `feature/RLAgent`

### Exact Prompts Submitted:

> "Can you add an attempt counter in the tetris game"

### AI Output Summary & Code Generated
I’ll add the attempt number to the visible game display, not just the terminal output. Since ALE’s `human` renderer does not expose a place to draw custom text, I’ll switch the display to a Pygame window that shows the Tetris frame with score and attempt information overlaid.

Read 

The current code already prints `Attempt` below the score in the terminal, but the ALE human window cannot display that custom text. I’m adding a Pygame HUD around the live frame so the attempt counter and current score are visible while the bot plays.

Generating patch (32 lines) in 


```

```


Generating patch (31 lines) in 


```

```
The HUD is now initialized once, refreshed after every environment step, and cleaned up on exit. I’m running syntax, diagnostics, and a short startup test to ensure the Pygame display works with the Atari RGB frames.

Ran terminal command: & "c:/Users/TryDo/Senior Project 2/.venv/Scripts/python.exe" -m py_compile .venv/ale.py

Checked , no problems found

Ran terminal command: $env:SDL_VIDEODRIVER='dummy'; & "c:/Users/TryDo/Senior Project 2/.venv/Scripts/python.exe" .venv/ale.py

Killed terminal

Added a visible Pygame HUD to `ale.py`.

The game window now displays:

```text
Score: 0
Attempt: 1
```

The attempt counter updates each time a new episode begins, and the score updates during gameplay.

The script compiles successfully and the HUD passed a runtime smoke test.

Made changes.

def update_game_display(observation, score, attempt, display, font):
    """Show the Atari frame with the current score and attempt counter."""
    frame = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))
    frame = pygame.transform.scale(frame, (frame.get_width() * 3, frame.get_height() * 3))
    display.fill((20, 20, 20))
    display.blit(frame, (0, 0))
    score_text = font.render(f"Score: {score:.0f}", True, (255, 255, 255))
    attempt_text = font.render(f"Attempt: {attempt}", True, (255, 220, 80))
    panel_y = frame.get_height()
    display.blit(score_text, (12, panel_y + 8))
    display.blit(attempt_text, (12, panel_y + 42))
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stop_requested.set()

### Human Testing, Verification & Results
The code was saved and ran once more to confirm that all previous functions remained the same, and the code added an attempt counter. The new score and attempt counter were added to the bottom of the tetris window in readable font and color.

---

## Entry 5: RL Agent - Afterstate Tetris Agent (Training, Watch, Controller Modes)

* **Developer:** Sam Culp
* **Date:** October 21-23
* **Tool Used:** Claude (Anthropic)
* **Associated Git Issue:** Implement Afterstate Learning (#27)
* **Associated Feature Branch:** `feature/afterstate`

### Key Prompts Submitted:

**Controller integration and hardware**

> "given these three files, we need to ensure arduino is being used to advance game state. first, see if there is a "FIRE" option instead of repeatedly holding down"

> "holding the down button makes the actuators really hot so i need a different solution"

> "https://ale.farama.org/env-spec/ view documentation. is there any way to quick fire so the pieces can zap to the bottom?"

> "assume I make it so once we are at the correct column, we dont press down at all. we simply let it fall. would this mean we need to retrain the model?"

> "assume I speed up gravity. would that require retraining?"

**Understanding and rebuilding the agent**

> "lets go ahead. make sure you are providing me with the code so i can write along and fully implement myself. also, quickly create a readme for this software portion before we begin"

> "where is the actual learning occurring and how are screens being scored? it has played 150 games and only cleared two lines"

> "So how does reward actually correlate to training? Where does loss function come in? How do we compare what we did vs what was expected"

> "so is the goal to move predicted towards target? why is target in this case necesasrily what we want"

> "lets create new files to separate watch mode and controller mode"

### AI Output Summary & Code Generated

**Part 1: Controller integration (`placement.py`, `main.py`).** I started with three existing files (`placement.py`, `afterstate.py`, `main.py`) and asked Claude to make the physical controller drive the emulator in Arduino mode. Before writing code, Claude asked a clarifying question. I answered that the Arduino presses real buttons through actuators and that the controller's USB reports should close the loop. Claude then made these changes:

* Added an `input_source` hook to `PlacementEnv` in `placement.py`. When it is set, the emulator is stepped with the action the physical controller reports instead of the action the executor requested. When it is `None` (the default), the original behavior is unchanged.
* Added a `ControllerInput` class in `main.py` that reads the pad through pygame, and a missed-pulse counter so I can see when the Arduino presses nothing the PC detects.
* Gated all of it to menu option 3, with setup before the env is modified and teardown in a `finally` block, so training and watch modes are unaffected.
* Delivered full replacement versions of `placement.py` and `main.py` at my request.

**Fast drop investigation.** Claude explained that FIRE is the rotate button, not a drop, and wrote a probe script that tries every action in ALE's full 18-action space and reports what each does to the piece. It could not run ALE itself. I then pointed Claude at the ALE documentation. Its findings: Tetris uses a 5-action set (NOOP, FIRE, RIGHT, LEFT, DOWN), the extra full-space actions are only joystick combinations, and nothing documents a hard drop.

**Actuator heat.** Holding DOWN made the actuators overheat. Claude added a `soft_drop` switch (`SOFT_DROP` in `main.py`) so DOWN is never pressed and gravity drops the piece, plus an optional duty-cycle setting (`DROP_RELEASE_FRAMES`, duty = 4 / (4 + idle frames)). It also suggested hardware-side fixes (full power briefly, then lower PWM to hold; a maximum on-time timer in the firmware). Claude confirmed that dropping DOWN presses does not require retraining, because the model only chooses a final orientation and column.

**Part 2: Rebuilding the agent from the ground up.** I wanted to understand how the files work. I started with only a basic understanding of reinforcement learning, including the afterstate approach they use. Claude explained the idea and walked through rebuilding the files from the ground up in small, ordered sections. I did not alter prior implementation in any significant way. Rather, I just completely rebuilt what Claude had generated and had it give me a deep dive on every step of implementation.

**Concept explained:** instead of making decisions every frame, the agent chooses one final placement per piece. The board that results from a placement is the "afterstate". A small neural network scores each candidate afterstate, and the agent plays the highest-scoring one (or a random one with probability epsilon, to explore). Learning is TD(0): each afterstate's score is moved toward `reward + discount * best next score`.

**Build order, with the code provided in each step:**

1. `placement.py`: reading the board from emulator pixels, finding the falling piece, piece shapes and rotations, simulating where a piece lands, and board features (holes, height, bumpiness).
2. `placement.py`: the closed-loop executor that presses buttons and checks the screen, plus the feature vector and replay buffer.
3. `afterstate.py`: the value network, the learner (TD target, loss, weight update), checkpoint saving, and the training loop.
4. `main.py`: the pygame display, menu, and training mode.
5. `watch.py` (watch mode) and `controller.py` (Arduino and USB controller mode), split into separate files at my request.
6. A README and `requirements.txt`.

Docstrings and comments were added throughout at my request so the code explains itself.

### Human Testing, Verification & Results

**Controller integration:** I ran controller mode on the physical hardware. That testing found the DOWN actuator overheating, the unclear mapping between the pad and the emulator, and the agent ramming pieces into the wall. The last was traced to the Arduino ID mapping once I supplied the firmware IDs. I checked Claude's statements against my files, for example when it said there is no gravity setting in them.

**Agent rebuild:** after each section I ran a small check script and compared the output to the expected values Claude gave. Examples of results I confirmed:

* A T piece on an empty board had 34 candidate placements, a flat I piece dropped to row 21, and a placement that completed the bottom row cleared 1 line.
* Board features for a column with one hole matched the expected values.
* The executor placed the first piece at the bottom-left of the board.
* The replay buffer returned the expected array shapes, and the network produced scores of the expected shape, trained, saved and reloaded a checkpoint.

Training mode then ran on the emulator. Watch mode was tested and slowed down with a frame-rate setting so the play could be followed.

I also questioned the results and the explanations (what is learned vs. fixed code, where the reward and loss are applied, why the target is a valid training signal). Claude traced each answer to specific lines in the code.

### Human Review & Modifications Identified

* **Wrong Arduino mapping.** Claude's first controller-mode code kept an identity `ARDUINO_MAP`, which did not match my firmware IDs and made the agent press the wrong buttons. This only surfaced on the hardware, and I had to supply the real IDs.
* **AI advice was wrong once.** Claude told me to set `SOFT_DROP = True` and described it as a hardware leftover that slowed training. Comparing against the original files showed training had used `False`, so that advice changed the behavior and was reverted. The code in all three files was otherwise confirmed to match the originals.
* **Piece sequence is fixed.** Claude's first suggestions to vary games (random idle frames, reseeding the emulator) did not work. A test I ran showed the emulator deals the same piece sequence every game, so greedy watch mode replays identical games. A small random-move probability in watch mode was added as a workaround.
* **Controller mode retest.** The first controller-mode version was tested on hardware as described above. The split-out `controller.py` has been written and tested on the hardware.
* I chose to split watch and controller modes into their own files and to keep `SOFT_DROP`, serial port, and pulse timing as settings in `controller.py`.
* On Windows, `pygame` failed to install on my Python version, so `requirements.txt` was changed to `pygame-ce`, which provides the same `pygame` module [confirm].