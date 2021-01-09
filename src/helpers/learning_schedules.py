# Adapted from FastAi's course part 2 lecture 9 on model training: https://course19.fast.ai/part2

import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import typing as t


def annealer(f: t.Callable):
    """
    Wraps a function requiring start, end and position parameters into a function that only requires the
    position parameter.
    """
    def _inner(start, end):
        return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start: float, end: float, x: float):
    return end + x * (start - end)


@annealer
def sched_no(start: float, end: float, x: float):
    return start


@annealer
def sched_cos(start: float, end: float, x: float):
    return start + (1 + math.cos(math.pi * x)) * (end - start) / 2


@annealer
def sched_exp(start: float, end: float, x: float):
    return start * (end / start)**(1 - x)


def combine_scheds(percents: t.List[float], scheds: t.List[t.Callable]):
    """Combines several schedule functions into a single one.

    Args:
        percents:  List of percentage representing the proportion of the training budget used by each scheduling
        function.
        scheds: List of scheduling functions accepting the proportion of remaining training time as input and providing
        the scheduled value as output.

    Returns:
        A scheduling function that is the concatenation of all provided functions, scaled by the percentages given.
        For example, if percents ares 0.4 and 0.6 then the first schedule will be followed for an amount equal to 40% of
        the budget and the second schedule will be followed for the remaining 60%.
    """

    assert np.sum(percents) == 1 and np.min(percents) >= 0
    pcts = np.cumsum(np.array([0] + percents))

    def _inner(x):
        elapsed = 1.0 - x
        idx = min(np.max(np.nonzero(elapsed >= pcts)), len(percents) - 1)
        actual_pos = (pcts[idx + 1] - elapsed) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner


if __name__ == '__main__':
    a = np.arange(0, 100)
    p = np.linspace(1.00, 0, 100)
    sched = combine_scheds([0.3, 0.7], [sched_lin(0.3, 0.6), sched_cos(0.6, 0.2)])
    plt.plot(a, [sched_lin(0.3, 0.6)(o) for o in p])
    plt.plot(a, [sched_cos(0.3, 0.6)(o) for o in p])
    plt.plot(a, [sched(o) for o in p])
    plt.show()
