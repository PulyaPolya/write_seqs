import typing as t

import pandas as pd
from music_df.augmentations import aug_by_trans, aug_rhythms, aug_within_range

from write_seqs.constants import HI_PITCH, LOW_PITCH
from write_seqs.settings import SequenceDataSettings


def augment(
    split: str,
    labeled_df: pd.DataFrame,
    settings: SequenceDataSettings,
    synthetic: bool,
) -> t.Iterator[pd.DataFrame]:
    apply_augment = (split == "train") and (
        (not synthetic) or settings.aug_synthetic_data
    )
    # we need to start by wrapping labeled_df in an iterable
    if not apply_augment:
        yield labeled_df
        return

    df_iter = iter([labeled_df])
    if settings.aug_by_key:
        assert not settings.aug_within_range
        df_iter = aug_by_trans(
            df_iter, settings.aug_by_key_n_keys, hi=HI_PITCH, low=LOW_PITCH
        )
    if settings.aug_within_range:
        assert not settings.aug_by_key
        df_iter = aug_within_range(df_iter, settings.aug_within_range_n_keys)
    if settings.aug_rhythms:
        df_iter = aug_rhythms(
            df_iter,
            settings.aug_rhythms_n_augs,
            n_possibilities=settings.aug_rhythms_n_possibilities,
        )
    yield from df_iter
