# RL Gaming Robot Software
This directory contains the software development files for the RL Gaming Robot project. It includes requirements, environment python codes, agent python codes, and test case codes.

## Directory Structure
```text

software/
├── requirements/
├── environment/
├── agent/
└── test_cases/

```
## Requirements
The requirements for the software side of the project must include a few main features
* Python 3.x - Main programming language
    * Gymnasium Library - Reinforcement Learning environment
    * PyTorch - Used to develop and train the RL model 
    * PyTorch Vision - Addon to PyTorch to handle video and images
    * PyInput - Used to listen to inputs while not having an open window like PyGame
* Git - Version control and repository management

## Environment
This project plans to use the gymnasium library to create an environment for the agent

## Agent
The agent is also provided via Gymnasium

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

## Test 4: Testing Gameplay Memory
This unit test just makes sure the new_memory function is correctly storing game play experience. Checking state, action, reward, next state, and game completion are all saved as one experience.
    Test code can be found [here:](tests/test_memory.py)

## Test 5: Testing LazyGamer Neural Network
This unit test is making sure that for every for every input there is the correct number of outputs.
    Test code can be found [here:](tests/test_lazygamer.py)

## Test 6: Testing Colorblind
This unit test is verifying that the colorblind fuction is converting the game image into the expected size, the expected type, and the correct coloring
    Test code can be found [here:](tests/test_colorblind.py)

## Test 7: Testing Action Selection
This unit test is making sure that the only actions being performed during runtime are only actions that are allowed and is testing the behavior when 'button mashing' happens.
    Test code can be found [here:](tests/test_foward.py)

## Test 8: Testing Environment Translation
This unit test is making sure that in the step_environment() function the program is correctly returning the commands, 'no action', 'down', 'left', 'right', and 'rotate'
    Test code can be found [here:](tests/test_step_environment.py)

## Test 9: Testing Arduino Commands
This unit test creates a mock arduino and records inputs allowing pytest to check the output. For this test to work, ou must uncomment the 'send_action_to_arduino' function
    Test code can be found [here:](tests/test_arduino.py)