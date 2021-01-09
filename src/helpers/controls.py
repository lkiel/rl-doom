import itertools
import typing as t

import numpy as np
from vizdoom.vizdoom import Button

ActionList = t.List[t.List[float]]

MUTUALLY_EXCLUSIVE_GROUPS = [
    [Button.MOVE_RIGHT, Button.MOVE_LEFT],
    [Button.TURN_RIGHT, Button.TURN_LEFT],
    [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
]


def get_available_actions(buttons: np.array, include_combinations=False, include_no_op=True) -> t.List[t.List[float]]:
    """Build the list of possible actions given authorized buttons.

    Each action is represented by a list of button states, encoded by float values. For example, if authorized buttons
    are 'ATTACK', 'MOVE_LEFT', 'MOVE_RIGHT', then the list [1.0, 0.0, 0.0] corresponds to 'ATTACK'. If combinations are
    enabled, then there can be more than one "enabled" button at once. In such case [1.0, 1.0, 0.0] would correspond
    to moving left AND attacking at the same time.

    Args:
        buttons: The list of available button for the environment.
        include_combinations: If true, the returned list will also contain valid combination of inputs.
        include_no_op: If true, the returned list will contain the all-zero action (corresponding to no action).
    Returns:
        A list of lists where the first dimension contains all possible actions and the second dimension all the button
        states of each action.
    """

    if not include_combinations:
        possible_actions = np.eye(len(buttons)).tolist()

        if include_no_op:
            possible_actions.append([0.] * len(buttons))

    else:
        # Create exclusion masks of size (n_available_buttons x n_exclusion_groups)
        mutual_exclusion_mask = np.array(
            [np.isin(buttons, excluded_group) for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS])

        # Create list of all possible actions of size (2^n_available_buttons x n_available_buttons)
        action_combinations = np.array([list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

        # Build action mask from action combinations and exclusion mask
        illegal_mask = np.any(np.sum(
            (action_combinations[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)), axis=-1) > 1,
                              axis=-1)

        possible_actions = action_combinations[~illegal_mask]

        if not include_no_op:
            possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]

    print('Built action space of size {} from buttons {}'.format(len(possible_actions), buttons))
    return possible_actions


if __name__ == '__main__':
    available_buttons = np.array([
        Button.ATTACK,
        Button.MOVE_FORWARD,
        Button.MOVE_LEFT,
        Button.MOVE_RIGHT,
        Button.TURN_LEFT,
        Button.TURN_RIGHT,
    ])

    actions = get_available_actions(available_buttons, include_no_op=False, include_combinations=True)
    print(actions)
