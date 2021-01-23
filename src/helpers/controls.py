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

EXCLUSIVE_BUTTONS = [
    Button.ATTACK,
]


def has_exclusive_button(actions: np.ndarray, buttons: np.array) -> np.array:
    """
    Identifies actions that should be excluded according to the exclusive list. If button A is in the exclusive list,
    then no other button can be active at the same time as A.

    Note that n_actions = 2^n_available_buttons.

        Args:
            actions: A binary array of shape (n_actions x n_available_buttons)
            buttons: The array of available buttons for the environment of shape (n_available_buttons).

        Returns:
            A boolean array of shape (n_actions) where entries set to true denote illegal actions as per the exclusion
            list.
        """

    exclusion_mask = np.isin(buttons, EXCLUSIVE_BUTTONS)
    return (np.any(actions.astype(bool) & exclusion_mask, axis=-1)) & (np.sum(actions, axis=-1) > 1)


def has_excluded_pair(actions: np.ndarray, buttons: np.array) -> np.array:
    """
    Identifies actions that should be excluded according to the pairwise exclusion groups. If buttons A and B are
    in the same exclusion group, then either A or B can be active at the same time but not both (logical XOR).

    Note that n_actions = 2^n_available_buttons.

    Args:
        actions: A binary array of shape (n_actions x n_available_buttons)
        buttons: The array of available buttons for the environment of shape (n_available_buttons).

    Returns:
        A boolean array of shape (n_actions) where entries set to true denote illegal actions as per the pairwise
        exclusion groups.
    """
    # Create mask of shape (n_mutual_exclusion_groups, n_available_buttons), marking location of excluded pairs.
    mutual_exclusion_mask = np.array([np.isin(buttons, excluded_group) for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS])

    # Flag actions that have more than 1 button active in any of the mutual exclusion groups.
    return np.any(
        np.sum(
            # Resulting shape (n_actions, n_mutual_exclusion_groups, n_available_buttons)
            (actions[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)),
            axis=-1
        ) > 1,
        axis=-1
    )


def get_available_actions(buttons: np.array, include_combinations=False, include_no_op=True) -> t.List[t.List[float]]:
    """Build the list of possible actions given authorized buttons.

    Each action is represented by a list of button states, encoded by float values. For example, if authorized buttons
    are 'ATTACK', 'MOVE_LEFT', 'MOVE_RIGHT', then the list [1.0, 0.0, 0.0] corresponds to 'ATTACK'. If combinations are
    enabled, then there can be more than one "enabled" button at once. In such case [1.0, 1.0, 0.0] would correspond
    to moving left AND attacking at the same time.

    Args:
        buttons: The array of available button for the environment.
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
        # Create list of all possible actions of size (2^n_available_buttons x n_available_buttons)
        action_combinations = np.array([list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

        # Build action mask from action combinations and exclusion mask
        illegal_mask = (has_excluded_pair(action_combinations, buttons)
                        | has_exclusive_button(action_combinations, buttons))

        possible_actions = action_combinations[~illegal_mask]

        if not include_no_op:
            possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]

    print('Built action space of size {} from buttons {}'.format(len(possible_actions), buttons))
    return possible_actions.tolist()


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
