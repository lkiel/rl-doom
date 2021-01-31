import random
import string

import numpy as np
from vizdoom.vizdoom import GameVariable

# TODO: empirically found, try to find official definition
from config import EnvironmentConfig
from environments.doom_env import DoomEnv

MAX_DISTANCE = 16.66
GAME_TIC_RATE = 35

AMMO_VARIABLES = [GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4,
                  GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9]

WEAPON_VARIABLES = [GameVariable.WEAPON0, GameVariable.WEAPON1, GameVariable.WEAPON2, GameVariable.WEAPON3,
                    GameVariable.WEAPON4,
                    GameVariable.WEAPON5, GameVariable.WEAPON6, GameVariable.WEAPON7, GameVariable.WEAPON8,
                    GameVariable.WEAPON9]


class DoomWithBots(DoomEnv):

    def __init__(self, doom_game, possible_actions, environment_config: EnvironmentConfig):
        super().__init__(doom_game, possible_actions, environment_config)

        self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
        self.n_bots = environment_config.env_args['bots']
        self.total_rew = 0
        self.last_damage_dealt = 0
        self.deaths = 0

        self.last_frags = 0
        self.last_health = 100
        self.last_armor = 0
        self.last_x, self.last_y = self._get_player_pos()
        self.ammo_state = self._get_ammo_state()
        self.weapon_state = self._get_weapon_state()

        self.rewards_stats = {
            'frag': 0,
            'damage': 0,
            'ammo': 0,
            'health': 0,
            'armor': 0,
            'distance': 0,
        }

        self.tic_rate = environment_config.frame_skip

        # TODO move to constants or config file
        # Rewards
        # 1 per kill
        self.reward_factor_frag = 1.0
        self.reward_factor_damage = 0.01

        # Player can move at ~16.66 units per tick
        self.reward_factor_distance = 0.00005
        self.penalty_factor_distance = 0.0025
        self.reward_threshold_distance = 3.0

        # Pistol clips have 10 bullets
        self.reward_factor_ammo_increment = 0.02
        self.reward_factor_ammo_decrement = -0.02

        # Player starts at 100 health
        self.reward_factor_health_increment = 0.02
        self.reward_factor_health_decrement = -0.01
        self.reward_factor_armor_increment = 0.01

        print(f'Logging with ID {self.name}')

        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')

    def shape_rewards(self, initial_reward: float):
        frag_reward = self._compute_frag_reward()
        damage_reward = self._compute_damage_reward()
        ammo_reward = self._compute_ammo_reward()
        health_reward = self._compute_health_reward()
        armor_reward = self._compute_armor_reward()
        distance_reward = self._compute_distance_reward(*self._get_player_pos())

        return initial_reward + frag_reward + damage_reward + ammo_reward + health_reward + armor_reward + distance_reward

    def _compute_distance_reward(self, x, y):
        dx = self.last_x - x
        dy = self.last_y - y

        self.last_x = x
        self.last_y = y

        # Proportion of max distance since last update between 0 and 1
        distance = np.sqrt(dx ** 2 + dy ** 2)

        d = distance - self.reward_threshold_distance

        if d > 0:
            reward = self.reward_factor_distance * d
        else:
            reward = self.penalty_factor_distance * d

        self._log_reward_stat('distance', reward)

        return reward

    def _compute_frag_reward(self):
        frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = self.reward_factor_frag * (frags - self.last_frags)

        self.last_frags = frags
        self._log_reward_stat('frag', reward)

        return reward

    def _compute_damage_reward(self):
        damage_dealt = self.game.get_game_variable(GameVariable.DAMAGECOUNT)
        reward = self.reward_factor_damage * (damage_dealt - self.last_damage_dealt)

        self.last_damage_dealt = damage_dealt
        self._log_reward_stat('damage', reward)

        return reward

    def _compute_health_reward(self):
        # When player is dead, the health game variable can be -999900
        health = max(self.game.get_game_variable(GameVariable.HEALTH), 0)

        health_reward = self.reward_factor_health_increment * max(0, health - self.last_health)
        health_penalty = self.reward_factor_health_decrement * min(0, health - self.last_health)
        reward = health_reward - health_penalty

        self.last_health = health
        self._log_reward_stat('health', reward)

        return reward

    def _compute_armor_reward(self):
        armor = self.game.get_game_variable(GameVariable.ARMOR)
        reward = self.reward_factor_armor_increment * max(0, armor - self.last_armor)
        self.last_armor = armor
        self._log_reward_stat('armor', reward)

        return reward

    def _compute_ammo_reward(self):
        self.weapon_state = self._get_weapon_state()

        new_ammo_state = self._get_ammo_state()
        ammo_diffs = (new_ammo_state - self.ammo_state) * self.weapon_state
        ammo_reward = self.reward_factor_ammo_increment * max(0, np.sum(ammo_diffs))
        ammo_penalty = self.reward_factor_ammo_decrement * min(0, np.sum(ammo_diffs))
        reward = ammo_reward - ammo_penalty
        self.ammo_state = new_ammo_state
        self._log_reward_stat('ammo', reward)

        return reward

    def _get_player_pos(self):
        return self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(
            GameVariable.POSITION_Y)

    def _get_ammo_state(self):
        ammo_state = np.zeros(10)

        for i in range(10):
            ammo_state[i] = self.game.get_game_variable(AMMO_VARIABLES[i])

        return ammo_state

    def _get_weapon_state(self):
        weapon_state = np.zeros(10)

        for i in range(10):
            weapon_state[i] = self.game.get_game_variable(WEAPON_VARIABLES[i])

        return weapon_state

    def _log_reward_stat(self, kind: str, reward: float):
        self.rewards_stats[kind] += reward

    def _reset_player(self):
        self.last_health = 100
        self.last_armor = 0
        self.game.respawn_player()
        self.last_x, self.last_y = self._get_player_pos()
        self.ammo_state = self._get_ammo_state()

    def _auto_change_weapon(self):
        # Determine the first weapon with ammo, starting from the most powerful ones TODO: make selection better
        possible_weapons = np.flatnonzero(self.ammo_state * self.weapon_state)
        possible_weapon = possible_weapons[-1] if len(possible_weapons) > 0 else None

        current_selection = self.game.get_game_variable(GameVariable.SELECTED_WEAPON)
        new_selection = possible_weapon if possible_weapon != current_selection else None

        return new_selection

    def step(self, action, array=False):
        # Apply action
        _ = self.game.make_action(self.possible_actions[action] if not array else action, self.frame_skip)
        reward = self.shape_rewards(initial_reward=0)

        self._respawn_if_dead()

        # Respawning takes a few ticks and might send us beyond the time limit.
        # Episode end needs to be checked last
        done = self.game.is_episode_finished()

        self.state = self.game_frame(done)
        self.total_rew += reward

        return self.state, reward, done, {'frags': self.last_frags}

    def reset(self):
        self._print_state()
        state = super().reset()

        self.last_x, self.last_y = self._get_player_pos()
        self.last_armor = 0
        self.last_health = 100
        self.last_frags = 0
        self.total_rew = 0
        self.deaths = 0

        # Damage count  is not cleared when starting a new episode: https://github.com/mwydmuch/ViZDoom/issues/399
        # self.last_damage_dealt = 0

        # Reset reward stats
        for k in self.rewards_stats.keys():
            self.rewards_stats[k] = 0

        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')

        return state

    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            # Check if player is dead
            if self.game.is_player_dead():
                self.deaths += 1
                self._reset_player()

    def _print_state(self):
        server_state = self.game.get_server_state()
        player_scores = list(zip(server_state.players_names, server_state.players_frags, server_state.players_in_game))
        player_scores = sorted(player_scores, key=lambda tup: tup[1])

        print('Results:')
        for player_name, player_score, player_ingame in player_scores:
            if player_ingame:
                print(f'{player_name}: {player_score}')
        print('************************')
        print('Agent {} frags: {}, deaths: {}, total reward: {}'.format(
            self.name,
            self.last_frags,
            self.deaths,
            self.total_rew
        ))
        for k, v in self.rewards_stats.items():
            print(f'- {k}: {v:+.1f}')
        print('************************')
