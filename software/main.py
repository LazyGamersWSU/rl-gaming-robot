"""
Launcher for the placement/afterstate Tetris agent.

Needs placement.py, afterstate.py, watch.py and controller.py next to it:
    python main.py        # menu: 1 = train, 2 = watch, 3 = play on the Arduino controller
(python afterstate.py still works too and trains directly.)
"""

import pygame

# --- GUI ---
SCREEN_WIDTH = 640   # window size in pixels
SCREEN_HEIGHT = 720
FPS = 60             # menu refresh rate

# Colors as (red, green, blue), each 0-255.
BG = (15, 15, 20)
ACCENT = (100, 220, 255)
TEXT = (230, 230, 230)
EXIT = (255, 180, 180)


class Display:
    """The pygame window: draws the game frame plus a status bar, and shows the menu."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tetris afterstate agent")
        # self.screen: the window's drawing surface.
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()  # used to cap the menu's frame rate
        # Fonts are created once; creating a font every frame is very slow.
        self.title_font = pygame.font.SysFont("consolas", 32)
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)

    def draw(self, rgb, mode, stats=None):
        """
        Draw one frame of the game plus a status bar.

        Args:
            rgb: ndarray (210, 160, 3) uint8, the emulator frame.
            mode: str, label shown in the status bar, e.g. "Afterstate".
            stats: dict mapping label (str) -> value, shown across the bar.
        """
        self.screen.fill(BG)

        # pygame wants arrays indexed [x, y], but frames are [y, x], so swap
        # axes 0 and 1, turn that into a pygame surface (an image), and stretch
        # it to fill the window above the status bar.
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (SCREEN_WIDTH, SCREEN_HEIGHT - 110))
        self.screen.blit(surface, (0, 0))  # blit = copy onto the screen at (x, y)

        y = SCREEN_HEIGHT - 100
        self.screen.blit(self.font.render(f"MODE: {mode.upper()}", True, ACCENT), (10, y))
        for i, (key, value) in enumerate((stats or {}).items()):
            self.screen.blit(
                self.small_font.render(f"{key}: {value}", True, TEXT),
                (10 + i * 150, y + 28),  # each stat sits 150 px right of the last
            )
        self.screen.blit(
            self.small_font.render("SPACE: Exit", True, EXIT), (10, SCREEN_HEIGHT - 25)
        )
        pygame.display.flip()  # show everything just drawn

    @staticmethod
    def should_exit():
        """Return True if the window was closed or SPACE was pressed."""
        # pygame collects keyboard/mouse/window events in a queue; get() drains it.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return True
        return False

    def choose_mode(self):
        """
        Show the menu and wait for a choice.

        Returns:
            "train", "watch" or "controller", or None if the user exits.
        """
        # (font, text, color, vertical offset from the window's center)
        lines = [
            (self.title_font, "Tetris afterstate agent", ACCENT, -100),
            (self.font, "1 - Train", TEXT, -30),
            (self.font, "2 - Watch trained agent", TEXT, 10),
            (self.font, "3 - Play on controller (Arduino)", TEXT, 45),
            (self.font, "SPACE - Exit", EXIT, 100),
        ]
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return None
                    if event.key == pygame.K_1:
                        return "train"
                    if event.key == pygame.K_2:
                        return "watch"
                    if event.key == pygame.K_3:
                        return "controller"

            self.screen.fill(BG)
            for font, text, color, dy in lines:
                surf = font.render(text, True, color)
                # get_rect(center=...) positions the text centered on that point.
                self.screen.blit(
                    surf,
                    surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + dy)),
                )
            pygame.display.flip()
            self.clock.tick(FPS)  # sleep so the loop runs at most FPS times/second


def main():
    """Open the window, show the menu, and run the chosen mode."""
    # Imported here because afterstate.py imports Display from this module.
    # Importing it at the top of this file would make each file wait on the other.
    from afterstate import Learner, train
    from controller_mode import play_controller
    from placement import PlacementEnv
    from watch_mode import watch

    display = Display()
    env = PlacementEnv()
    learner = Learner()

    try:
        mode = display.choose_mode()  # "train", "watch", "controller" or None
        if mode is None:
            return

        loaded = learner.load()  # True if a checkpoint was found and restored
        if mode in ("watch", "controller"):
            # Both modes need a trained network to play with.
            if not loaded:
                print("No checkpoint (tetris_afterstate.pt) found. Train first.")
                return
            if mode == "watch":
                watch(display, env, learner)
            else:
                play_controller(display, env, learner)
        else:
            print("Resumed from checkpoint." if loaded else "Starting fresh.")
            try:
                train(display, env, learner)
            except KeyboardInterrupt:  # Ctrl+C in the terminal saves progress
                learner.save()
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()