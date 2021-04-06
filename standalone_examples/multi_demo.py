import sys

import vizdoom

sys.path.insert(0, '../src')  # To make pretrained models work (module structure different than notebooks)

print("P1")
game = vizdoom.DoomGame()
game.load_config(f'scenarios/bots_deathmatch_multimaps.cfg')
game.set_doom_map('M')
game.set_mode(vizdoom.Mode.SPECTATOR)
game.add_game_args("-host 2 -port 12345 -deathmatch +cl_run 1 +name Human")

game.set_ticrate(35)

game.init()

for i in range(7):
    game.send_game_command("addbot")

while not game.is_episode_finished():
    game.advance_action()

    if game.is_player_dead():
        game.respawn_player()

game.close()
