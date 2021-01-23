import random

import vizdoom

if __name__ == '__main__':
    game = vizdoom.DoomGame()
    game.load_config('scenarios/basic.cfg')
    game.init()

    sample_actions = [
        [1, 0, 0],  # Move left
        [0, 1, 0],  # Move right
        [0, 0, 1],  # Attack
    ]

    n_episodes = 10
    current_episode = 0

    while current_episode < n_episodes:
        pass

    game.close()
