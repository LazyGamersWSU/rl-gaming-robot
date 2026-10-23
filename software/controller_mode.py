"""
Controller mode: the trained network chooses placements, an Arduino presses a
real controller, and the controller's USB state is what steps the emulator.

Started from main.py (menu option 3). Needs a saved checkpoint, an Arduino on
SERIAL_PORT, and a USB controller. Requires pyserial.
"""
import json
import time

import numpy as np
import pygame

from placement import NOOP, RIGHT, LEFT, ExitRequested, candidates, featurize

FPS = 60  # emulator frames per second to hold the game to (matches a real console)

# --- Hardware ---
SERIAL_PORT = "COM3"          # str: the Arduino's port. Change to match yours.
SERIAL_BAUD = 115200          # int: serial speed; must match the Arduino sketch
SERIAL_STARTUP_DELAY = 2.0    # float: seconds to wait after opening the port (the Arduino resets)
INPUT_LOG_PATH = "input_log.json"  # where the record of sent pulses is written on exit

# Emulator action number -> id sent to the Arduino.
# Emulator: 0=NOOP, 1=FIRE (rotate), 2=RIGHT, 3=LEFT, 4=DOWN.
# Firmware: 1=A (rotate), 2=Y (down), 3=right, 4=left.
ARDUINO_MAP = {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}

# Every move/rotation/soft-drop is ONE pulse command; the Arduino decides how
# long to press. These add idle emulator frames (1 frame = 16.7 ms) after each
# pulse so the emulator never pulses faster than the hardware can.
MOVE_RELEASE_FRAMES = 2   # idle frames after a left/right/rotate pulse

# DOWN actuator heat control (applies in controller mode only).
SOFT_DROP = False         # False = never press DOWN; gravity drops the piece
DROP_RELEASE_FRAMES = 8   # only used if SOFT_DROP is True: idle frames after each DOWN pulse (DOWN duty cycle is 4 / (4 + DROP_RELEASE_FRAMES))

# Physical controller as pygame sees it (SNES pad over USB):
#   buttons: X=0, A=1, B=2, Y=3, Select=8, Start=9
#   D-pad: axis 0 (left = -1, right = +1); rests near -0.01, so use a dead zone
CONTROLLER_INDEX = 0      # int: which connected controller to read (0 = first)
BUTTON_ACTIONS = {        # dict: pygame button number -> emulator action
    1: 1,   # A -> FIRE (rotate)
    3: 4,   # Y -> DOWN
}
AXIS_THRESHOLD = 0.5      # float: how far the D-pad axis must move to count as a press


class ArduinoLink:
    """
    Sends button-pulse commands to the Arduino over serial. Each command is the
    text "<id>\\n". Nothing is held or released from this side: the Arduino
    decides how long each pulse lasts. Also slows the emulator to real time
    (FPS frames/s) so gravity matches a real console.
    """

    def __init__(self):
        import serial  # pyserial; imported here so other modes don't need it

        # self.port: the open serial connection. timeout=1 is seconds before a read gives up.
        self.port = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(SERIAL_STARTUP_DELAY)  # opening the port resets the Arduino; let it boot
        self.log = []  # list of dicts, one per pulse sent
        self.next_tick = time.perf_counter()  # float: when the next frame is due

    def pulse(self, action):
        """Send one pulse. action: int, an emulator action (0-4)."""
        self.port.write(f"{ARDUINO_MAP[action]}\n".encode("utf-8"))  # str -> bytes
        self.port.flush()  # send now rather than wait in a buffer
        self.log.append({"time": time.time(), "action": int(action)})

    def pace(self):
        """Sleep until the next 1/FPS-second frame boundary."""
        self.next_tick += 1 / FPS
        delay = self.next_tick - time.perf_counter()  # float: seconds to wait
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.1:  # more than 0.1 s behind (e.g. a slow draw): reset the
            self.next_tick = time.perf_counter()  # schedule instead of racing to catch up

    def close(self):
        """Close the serial port and write the pulse log to INPUT_LOG_PATH."""
        self.port.close()
        with open(INPUT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2)


class ControllerInput:
    """
    Reads the physical controller over USB through pygame. Calling an instance
    returns the emulator action matching what is pressed right now.
    """

    def __init__(self):
        pygame.joystick.init()
        if pygame.joystick.get_count() <= CONTROLLER_INDEX:
            raise RuntimeError("no USB controller detected by pygame")
        self.joy = pygame.joystick.Joystick(CONTROLLER_INDEX)  # the controller object
        self.joy.init()
        print(
            f"Controller: {self.joy.get_name()} | buttons {self.joy.get_numbuttons()} "
            f"hats {self.joy.get_numhats()} axes {self.joy.get_numaxes()}"
        )

    def __call__(self):
        """
        Returns:
            int: an emulator action (NOOP, FIRE, RIGHT, LEFT or DOWN).
            Buttons are checked first, then the D-pad; the first match wins.
        """
        pygame.event.pump()  # let pygame refresh the controller's state
        j = self.joy
        for b, action in BUTTON_ACTIONS.items():
            if b < j.get_numbuttons() and j.get_button(b):  # 1 if pressed
                return action
        if j.get_numaxes() >= 1:
            ax = j.get_axis(0)  # float, -1.0 (left) to +1.0 (right)
            if ax > AXIS_THRESHOLD:
                return RIGHT
            if ax < -AXIS_THRESHOLD:
                return LEFT
        return NOOP


def play_controller(display, env, learner):
    """
    Play with the trained network through real hardware, game after game, until
    the user exits. The executor sends pulse commands to the Arduino, which
    presses the real controller; the USB controller's state steps the emulator.

    Args:
        display: Display, the pygame window.
        env: PlacementEnv, the emulator wrapper.
        learner: Learner, with a checkpoint already loaded.
    """
    # Setup first: if it fails, the env hasn't been modified at all.
    try:
        controller = ControllerInput()
        link = ArduinoLink()
    except Exception as exc:  # serial error, pyserial missing, no controller, ...
        print(f"Controller mode setup failed: {exc}")
        return

    stats = {"Pieces": 0, "Lines": 0, "Missed": 0}  # dict shown in the status bar
    frames = [0]  # frame counter; a list so on_frame can modify it

    def on_frame(action):
        """Called by env after every emulator frame: draw, check exit, pace."""
        frames[0] += 1
        if frames[0] % 4 == 0:  # drawing every 4th frame is enough
            display.draw(env.obs, "Controller", stats)
            if display.should_exit():
                raise ExitRequested
        link.pace()  # hold the emulator to real time

    # Remember the env's normal settings so we can restore them afterwards.
    saved = (env.release, env.drop_release, env.soft_drop)

    # Switch the env into controller state.
    env.input_source = controller      # controller state overrides executor actions
    env.on_pulse = link.pulse          # each executor press also goes to the Arduino
    env.on_frame = on_frame
    env.release = MOVE_RELEASE_FRAMES
    env.drop_release = DROP_RELEASE_FRAMES
    env.soft_drop = SOFT_DROP
    print("controller mode ON | soft_drop =", env.soft_drop, "| drop_release =", env.drop_release)
    try:
        while True:  # one pass = one game
            env.reset()
            env.missed = 0  # pulses the controller never reported
            stats["Pieces"] = stats["Lines"] = stats["Missed"] = 0
            while not env.done:
                state = env.state()
                cands = candidates(state[2], state[0]) if state is not None else []
                if not cands:
                    break
                feats = np.stack([featurize(after, c) for _, _, after, c in cands])
                shape, col, _, cleared = cands[int(np.argmax(learner.values(feats)))]
                env.play(shape, col)
                stats["Pieces"] += 1
                stats["Lines"] += cleared
                stats["Missed"] = env.missed
            print(
                f"Game over: {stats['Pieces']} pieces, {stats['Lines']} lines, "
                f"{env.missed} missed pulses"
            )
    except ExitRequested:
        pass
    finally:
        # Leave controller state: put the env back as it was.
        env.input_source = None
        env.on_pulse = env.on_frame = None
        env.release, env.drop_release, env.soft_drop = saved
        link.close()
        print("controller mode OFF")