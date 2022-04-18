import numpy as np
import vizdoom
from vizdoom.vizdoom import ScreenResolution


def get_sample_frame(resolution: ScreenResolution) -> np.ndarray:
    # Load game
    game = vizdoom.DoomGame()
    game.load_config('scenarios/basic.cfg')
    game.set_seed(0)
    game.set_screen_resolution(resolution)
    game.init()

    # Collect frame
    frame = game.get_state().screen_buffer

    game.close()

    return frame
