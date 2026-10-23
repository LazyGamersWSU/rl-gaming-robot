# Controller Input Mapping Experiment

## Overview

The purpose of this experiment is to record what python sees when a button is pressed on the USB SNES controller.

## Results

Controller input testing was completed using the Python `pygame` experiment.

### Button Mapping

| SNES Button | Input |
|---|---:|
| X | Button 0 |
| A | Button 1 |
| B | Button 2 |
| Y | Button 3 |
| Select | Button 8 |
| Start | Button 9 |

### D-Pad Mapping

The D-pad is detected as two joystick axes rather than individual buttons or a hat.

| Direction | Axis | Value When Pressed |
|---|---:|---:|
| Left | Axis 0 | -1 |
| Right | Axis 0 | 1 |
| Up | Axis 1 | -1 |
| Down | Axis 1 | 1 |
| Released | Axis 0/1 | -0.01 |

### Notes

- Horizontal D-pad input is reported on Axis 0.
- Vertical D-pad input is reported on Axis 1.
- D-pad values are `-1` or `1` when pressed.
- When released, the controller reports approximately `-0.01` instead of exactly `0`, so input handling should use a dead zone rather than checking for an exact value of zero.