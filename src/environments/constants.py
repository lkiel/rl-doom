from vizdoom.vizdoom import ScreenFormat, GameVariable, Mode
from vizdoom.vizdoom import ScreenResolution

# Mapping between the number of channels used in the input and the corresponding VizDoom screen format.
CHANNELS_TO_VZD = {1: ScreenFormat.GRAY8, 3: ScreenFormat.RGB24}

MODE_TO_VZD = {
    "PLAYER": Mode.PLAYER,
    "ASYNC_PLAYER": Mode.ASYNC_PLAYER,
    "SPECTATOR": Mode.SPECTATOR,
    "ASYNC_SPECTATOR": Mode.ASYNC_SPECTATOR,
}

# Mapping between tuple of (width, height) and the corresponding VizDoom screen resolution.
RESOLUTION_TO_VZD = {(320, 240): ScreenResolution.RES_320X240}

CHANNELS = 3
CROP_TOP = 40
OBS_SHAPE = (100, 160, CHANNELS)
SCREEN_SHAPE = (240, 320, CHANNELS)

AMMO_VARIABLES = [
    GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4,
    GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9
]

WEAPON_VARIABLES = [
    GameVariable.WEAPON0, GameVariable.WEAPON1, GameVariable.WEAPON2, GameVariable.WEAPON3, GameVariable.WEAPON4,
    GameVariable.WEAPON5, GameVariable.WEAPON6, GameVariable.WEAPON7, GameVariable.WEAPON8, GameVariable.WEAPON9
]
