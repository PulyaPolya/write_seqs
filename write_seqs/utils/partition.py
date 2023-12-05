import itertools as it
import math
import random
import typing as t
from bisect import bisect_right

# def partition(
#     proportions: t.Sequence[float], items: t.List[t.Any]
# ) -> t.Tuple[t.List[t.Any], ...]:
#     """`items` is shuffled in-place."""
#     random.shuffle(items)
#     breakpoint()
#     int_ratios = [int(round(x * len(items))) for x in proportions]
#     boundaries = list(it.accumulate(int_ratios))
#     boundaries[-1] = len(items)  # just in case it has been rounded down
#     return tuple(
#         items[start:stop] for (start, stop) in zip([0] + boundaries, boundaries)
#     )

T = t.TypeVar("T")


def partition(
    proportions: t.Sequence[float],
    items: t.List[T],
    item_lengths: t.List[int | float],
    shuffle: bool = True,
) -> t.Tuple[t.List[T], ...]:
    """`items` is shuffled in-place.
    >>> items = list(range(10))
    >>> item_lengths = [2] * 8 + [8] * 2
    >>> partition((0.5, 0.5), items, item_lengths, shuffle=False)
    ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    >>> partition((0.25, 0.25, 0.25, 0.25), items, item_lengths, shuffle=False)
    ([0, 1, 2, 3], [4, 5, 6, 7], [8], [9])
    >>> partition((0.0, 1.0), items, item_lengths, shuffle=False)
    ([], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> partition((1.0, 0.0), items, item_lengths, shuffle=False)
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [])

    >>> items = list(range(1000))
    >>> item_lengths = [random.randrange(10, 200) for _ in range(1000)]
    >>> proportions = (0.2, 0.5, 0.3)
    >>> splits = partition(proportions, items, item_lengths)
    >>> split_lengths = [[item_lengths[x] for x in split] for split in splits]
    >>> actual_proportions = [
    ...     sum(split_length) / sum(item_lengths) for split_length in split_lengths
    ... ]
    >>> assert all(
    ...     math.isclose(p, actual_p, abs_tol=0.01)
    ...     for p, actual_p in zip(proportions, actual_proportions)
    ... )
    >>> actual_proportions  # doctest: +SKIP
    [0.19958496787765082, 0.4988439744537306, 0.30157105766861864]


    """
    if shuffle:
        indices = list(range(len(items)))
        random.shuffle(indices)

        items = [items[i] for i in indices]
        item_lengths = [item_lengths[i] for i in indices]

    cumulative_lengths = list(it.accumulate(item_lengths))
    total_length = sum(item_lengths)
    thresholds = list(it.accumulate(x * total_length for x in proportions))
    boundaries = [
        bisect_right(cumulative_lengths, threshold) for threshold in thresholds
    ]
    return tuple(
        items[start:stop] for (start, stop) in zip([0] + boundaries, boundaries)
    )
