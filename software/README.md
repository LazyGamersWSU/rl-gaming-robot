# RL Gaming Robot Software
This directory contains the software development files for the RL Gaming Robot project.

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
Test cases are recorded and graphed/stored, showing that the RL model is improving.

# Version 1
9/18/2026
Currently the RL Agent has a Deep Q-Network that stores decisions and results into files. Pytorch sends the DQN game data and gives response to actions taken by the DQN. Ale.py houses the code to call the DQN and creates a tetris enviroment. The bot can interact with the game, winning one seems to be rare thus far. The commands for manual controls are interchangable with the bot control.

Currently it does not send out USB commands, and learning is iffy to prove.