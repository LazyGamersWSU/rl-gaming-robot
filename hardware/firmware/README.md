# Firmware

Arduino firmware for controlling the solenoid driver PCB and translating serial commands received over USB into physical controller inputs.

## Pin Assignments

| Arduino Pin | Solenoid | Function |
|---|---|---|
| 2 | A | Rotate |
| 3 | Y | Down |
| 4 | Right | D-pad right |
| 5 | Left | D-pad left |

## Serial Communication Protocol

The firmware communicates over USB serial at **115200 baud**. Each command is a single character (`'1'`–`'4'`) corresponding to a solenoid. Upon receiving a valid command, the Arduino activates the associated solenoid for 100 ms before deactivating it.

| Command | Solenoid | Action |
|---|---|---|
| `1` | A | Rotate |
| `2` | Y | Down |
| `3` | Right | D-pad right |
| `4` | Left | D-pad left |

## Emulator Action Mapping

The Python emulator uses a different action index ordering than the firmware. `ARDUINO_MAP` translates emulator actions into the corresponding firmware command IDs.

| Emulator Action Index | Emulator Action | Firmware Command | Physical Action |
|---|---|---|---|
| 0 | NOOP | 0 | No action |
| 1 | FIRE (rotate) | 1 | A — Rotate |
| 2 | RIGHT | 3 | Right |
| 3 | LEFT | 4 | Left |
| 4 | DOWN | 2 | Y — Down |

The mapping is defined in Python as:

```python
ARDUINO_MAP = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 2
}
```

Command `0` represents no action and does not trigger a solenoid pulse.

## Solenoid Control Logic

- All solenoid pins are configured as outputs and initialized to `LOW` at startup.
- The Arduino reads incoming serial commands and activates the corresponding solenoid.
- Each activation lasts 100 ms, after which the output returns to `LOW`.
- Unrecognized commands are ignored.