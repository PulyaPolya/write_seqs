import random
import typing as t

import pandas as pd
from music_df import chromatic_transpose

from write_seqs.constants import HI_PITCH, LOW_PITCH
from write_seqs.settings import SequenceDataSettings


def aug_within_range(
    df_iter: t.Iterable[pd.DataFrame],
    n_keys: int,
    hi: int = HI_PITCH,
    low: int = LOW_PITCH,
    min_trans: int = -12,
    max_trans: int = 12,
):
    # if n_keys is None, we transpose to every step within range
    avail_range = hi - low
    for df in df_iter:
        actual_max = int(df.pitch.max())
        actual_min = int(df.pitch.min())
        actual_range = actual_max - actual_min
        n_trans = avail_range - actual_range + 1
        if n_trans <= 0:
            continue
        trans = list(
            range(
                max(low - actual_min, min_trans),
                min(max_trans, low - actual_min + n_trans),
            )
        )
        if n_keys < n_trans:
            random.shuffle(trans)
            trans = trans[:n_keys]
        for t in trans:
            yield chromatic_transpose(df, t, inplace=False, label=True)


def aug_rhythms(
    df_iter: t.Iterable[pd.DataFrame],
    n_augs: int,
    threshold: float = 0.6547667782160375,
    metadata: bool = True,
):
    # default threshold was empirically calculated from the 177 scores that
    # are presently included when running with --all
    for df in df_iter:
        mean_dur = (df.release - df.onset).mean()
        pows_of_2 = [
            x - (n_augs // 2) + (mean_dur < threshold and n_augs % 2 == 0)
            for x in range(n_augs)
        ]
        for pow_of_2 in pows_of_2:
            if not pow_of_2:
                yield df
            else:
                scale_factor = 2.0**pow_of_2
                aug_df = df.copy()
                aug_df["onset"] *= scale_factor
                aug_df["release"] *= scale_factor
                # aug_df["durs_x"] = scale_factor
                if metadata:
                    if "rhythms_scaled_by" in aug_df.attrs:
                        aug_df.attrs["rhythms_scaled_by"] *= scale_factor
                    else:
                        aug_df.attrs["rhythms_scaled_by"] = scale_factor
                yield aug_df


def augment(
    split: str,
    labeled_df: pd.DataFrame,
    settings: SequenceDataSettings,
    synthetic: bool,
) -> t.Iterable[pd.DataFrame]:
    augment = (split == "train") and ((not synthetic) or settings.aug_synthetic_data)
    # we need to start by wrapping labeled_df in an iterable
    df_iter = [labeled_df]
    if not augment:
        return df_iter
    if settings.aug_within_range:
        df_iter = aug_within_range(df_iter, settings.aug_within_range_n_keys)
    if settings.aug_rhythms:
        df_iter = aug_rhythms(df_iter, settings.aug_rhythms_n_augs)
    return df_iter
