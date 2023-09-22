import itertools as it
import random
import typing as t


def partition(
    proportions: t.Sequence[float], items: t.List[t.Any]
) -> t.Tuple[t.List[t.Any], ...]:
    """`items` is shuffled in-place."""
    random.shuffle(items)
    int_ratios = [int(round(x * len(items))) for x in proportions]
    boundaries = list(it.accumulate(int_ratios))
    boundaries[-1] = len(items)  # just in case it has been rounded down
    return tuple(
        items[start:stop] for (start, stop) in zip([0] + boundaries, boundaries)
    )
