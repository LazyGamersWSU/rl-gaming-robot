# Controller Latency Experiment

## Overview

Measured end-to-end latency between sending a command to the Arduino and detecting the resulting physical controller input. Python sends serial commands to the Arduino, which activates the corresponding solenoid on the controller. Pygame detects the resulting button or D-pad input and records the elapsed time.

## Results

Repeated trials demonstrated an end-to-end latency of approximately **40–55 ms** per command.

This measurement includes serial communication, Arduino processing, solenoid actuation, physical controller input, and input detection by Pygame.

## Purpose

Establish a baseline for physical input latency to inform the reinforcement learning agent's interaction with the physical gaming robot.