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
    * PyToch Vision - Addon to PuTorch to handle video and images
    * PyInput - Used to listen to inputs while not having an open window like PyGame
* Git - Version control and repository management

## Environment
This project plans to use the gymnasium library to create an environment for the agent

## Agent
The agent is also provided via Gymnasium

## Test Cases
All test cases are recorded and written here.

### Test 1: Simple Interactions With Game
This test is pretty straight forward, after starting the ale.py file, I manually controlled the game to make sure all inputs were being tracked. This being up, down, left, right, and both rotates. Once the game ended I went into the .json file that tracked all moves made and ctrl+f 'keyboard' to make the search easier. This showed me that all moves are being tracked properly. All moves were recorded and in the correct order, the timestamps were increased, and were accurate to the buttons I pressed (no misinputs).

### Test 2: 