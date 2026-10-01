# RL Gaming Robot

The RL Gaming Robot is a senior design project focused on developing a Reinforcement Learning system that learns to play games through repeated attempts and applies its learned behavior to a game to ultimately improve its score.

The current target game is Tetris. The long-term goal is for the RL agent to make gameplay decisions and output controller commands that are translated into physical actions pressed on a controller by a robotic system.

This repository contains the software and hardware elements used to create the project.

## Repository Structure

```text
RL-Gaming-Robot/

├── hardware/
│   ├── electronics/
│   ├── experiments/
│   ├── firmware/
│   ├── mechanical/
│   └── photos/
│ 
├── software/
│   ├── requirements/
│   ├── environment/
│   ├── agent/
│   └── test_cases/
│
└── README.md
```

## Hardware

The `hardware` directory contains the mechanical and electronic development files for the RL Gaming Robot.

This includes:

* Actuator experiments and testing
* CAD models and mechanical designs
* Controller mounting and interaction components
* Production hardware designs
* Hardware documentation and validation

See the [hardware README](hardware/README.md) for additional information.

## Software

The `software` directory contains the software required to develop and operate the RL system.

This includes:

* Python environment and dependency requirements
* Game environment code
* Reinforcement Learning agent code
* Software test cases

The software is responsible for interacting with the game environment, training and executing the RL agent, and producing controller actions for the hardware system.

See the [software README](software/README.md) for additional information.

## System Overview

The intended system operates as a feedback loop:

```text
┌─────────────────┐
│ Game Environment│
└────────┬────────┘
         │
         │ Game State
         ▼
┌─────────────────┐
│    RL Agent     │
└────────┬────────┘
         │
         │ Controller Action
         ▼
┌─────────────────┐
│ Robot Controller│
└────────┬────────┘
         │
         │ Physical Command
         ▼
┌─────────────────┐
│  Robotic System  │
└────────┬────────┘
         │
         │ Button Press
         ▼
┌─────────────────┐
│  NES Controller │
└────────┬────────┘
         │
         └──────────────► Game
```

The game produces a new state after each action, allowing the RL agent to continually evaluate the game and select its next action.

## Testing

Testing is an integral part of the project and is performed throughout both software and hardware development.

Software testing focuses on:

* Game-state processing
* RL agent behavior
* Action generation
* Software component integration
* Model performance and repeatability

Hardware testing focuses on:

* Actuator performance
* Button press accuracy
* Timing and reliability
* Mechanical positioning
* Hardware/software integration

Test cases are maintained within the appropriate software and hardware directories.

## Development Philosophy

The project is organized around **experimentation, validation, and integration**.

Development follows these general principles:

1. **Experiment** — Develop and test individual concepts and components.
2. **Validate** — Verify that components meet their intended requirements.
3. **Integrate** — Combine validated software and hardware components.
4. **Test** — Evaluate the complete system and identify integration issues.
5. **Iterate** — Use test results to improve the system.

Experimental implementations should be validated before being incorporated into the production system.

## Project Goals

The primary goals of the RL Gaming Robot project are to:

* Develop an RL agent capable of improving its gameplay through repeated attempts.
* Create a software environment for training and evaluating the agent.
* Develop a robotic system capable of physically interacting with a gaming controller.
* Establish reliable communication between the software and hardware systems.
* Test the performance and reliability of the complete system.
* Demonstrate the transfer of learned behavior from the software environment to physical gameplay.

## Current Development Status

The project is being developed incrementally, beginning with the software RL environment and agent before progressing toward full hardware integration.

Major development stages include:

1. Software environment development
2. RL agent development and training
3. Software testing and performance evaluation
4. Hardware development and validation
5. Software/hardware integration
6. Full-system testing
7. Final NES Tetris demonstration

## Team Roles

### Trystan Donmoyer - Software Development Lead
- Leads development of the reinforcement learning software and agent
- Develops training, inference, and game interaction systems
- Maintains the software architecture and coordinates software-side implementation

### Sam Culp - Hardware Development
- Leads hardware design and implementation of the physical gaming robot
- Designs and assembles the PCB, power distribution, actuator system, and mechanical components
- Develops and integrates Arduino firmware for hardware control and communication with the software system

### Matthew Dynes-Winberry - Integration & Testing
- Integrates the software and hardware systems into a complete working platform
- Develops and manages system-level testing and validation
- Troubleshoots issues across hardware/software interfaces and verifies system reliability