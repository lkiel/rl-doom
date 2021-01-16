import numpy as np
import vizdoom
from vizdoom.vizdoom import ScreenResolution


def get_sample_frame(resolution: ScreenResolution) -> np.ndarray:
    # Load game
    game = vizdoom.DoomGame()
    game.load_config('scenarios/basic.cfg')
    game.set_seed(0)

    # Test 160 by 120 resolution
    game.set_screen_resolution(resolution)
    game.init()

    frame = game.get_state().screen_buffer

    game.close()

    return frame
