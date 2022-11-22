import csv
import json
import os
import pickle
import random
import typing as t
import warnings

from dataclasses import dataclass
from numbers import Number

import pandas as pd

from reprs.midi_like import (
    MidiLikeSettings,
    midilike_encode,
    inputs_vocab_items,
)

from write_chord_tones_seqs.augmentations import augment
from write_chord_tones_seqs.utils.partition import partition
from write_chord_tones_seqs.settings import (
    ChordTonesDataSettings,
    save_dclass,
    path_from_dataclass,
)


@dataclass
class CorpusItem:
    csv_path: str
    synthetic: bool = False


def vocab_paths(output_folder):
    return (
        os.path.join(output_folder, "inputs_vocab.list.{ext}"),
        os.path.join(output_folder, "targets_vocab.list.{ext}"),
    )


def get_data_dir(output_folder: str):
    return os.path.join(output_folder, "data")


def get_split_dir(output_folder: str, split: str) -> str:
    input_path = os.path.join(get_data_dir(output_folder), split)
    return input_path


def get_items_from_corpora(
    src_data_dir: str,
    ct_settings: ChordTonesDataSettings,
    repr_settings: MidiLikeSettings,
) -> t.Tuple[t.List[str], t.List[str]]:
    _, corpora, _ = next(os.walk(src_data_dir))
    for corpus_name in ct_settings.corpora_to_exclude:
        if corpus_name not in corpora:
            warnings.warn(
                f"corpus '{corpus_name}' in `corpora_to_exclude` not recognized"
            )
    for corpus_name in ct_settings.training_only_corpora:
        if corpus_name not in corpora:
            warnings.warn(
                f"corpus '{corpus_name}' in `training_only_corpora` not recognized"
            )
    training_only_items = []
    items = []

    for corpus_name in corpora:
        if (
            ct_settings.corpora_to_include
            and corpus_name not in ct_settings.corpora_to_include
        ):
            continue
        if corpus_name in ct_settings.corpora_to_exclude:
            continue
        if (
            corpus_name in ct_settings.training_only_corpora
            and corpus_name not in ct_settings.corpora_to_include
        ):
            to_extend = training_only_items
        else:
            to_extend = items
        corpus_dir = os.path.join(src_data_dir, corpus_name)
        with open(os.path.join(corpus_dir, "attrs.json")) as inf:
            corpus_attrs = json.load(inf)
        if (
            repr_settings.include_metric_weights
            and not corpus_attrs["has_weights"]
        ):
            print(f"Corpus {corpus_name} has no weights, skipping")
            continue
        csv_paths = [
            os.path.join(corpus_dir, p)
            for p in os.listdir(corpus_dir)
            if p.endswith(".csv")
        ]
        if (
            prop := ct_settings.corpora_sample_proportions.get(
                corpus_name, None
            )
            is not None
        ):
            csv_paths = random.sample(csv_paths, int(prop * len(csv_paths)))
        to_extend.extend(
            [
                CorpusItem(csv_path, corpus_attrs["synthetic"])
                for csv_path in csv_paths
            ]
        )
    return items, training_only_items


def get_items(
    src_data_dir: str,
    ct_settings: ChordTonesDataSettings,
    repr_settings: MidiLikeSettings,
    seed: t.Optional[int] = None,
    proportions: t.Tuple[Number] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
    # corpora_to_exclude: t.Sequence[str] = (),
    # training_only_corpora: t.Sequence[str] = "synthetic",
) -> t.Tuple[t.List[CorpusItem], t.List[CorpusItem], t.List[CorpusItem]]:
    # We would like to know how many tokens each input file has in order to
    #   partition the splits as exactly as possible. But that would require
    #   processing them all first and for various reasons we don't want to do
    #   that (e.g., we want to apply augmentation to the training data
    #   in the middle of the processing pipeline). We could use the size of the
    #   files (as I did in midi_data) as a rough heuristic for how many notes
    #   they contain but since the file formats are heterogeneous this seems
    #   unlikely to be reliable. So we just hope the magic of randomness
    #   will give us a fairly even split.
    if seed is not None:
        random.seed(seed)
    items, training_only_items = get_items_from_corpora(
        src_data_dir, ct_settings, repr_settings
    )
    if frac != 1.0:
        items, _ = partition((frac, 1.0 - frac), items)
        training_only_items, _ = partition(
            (frac, 1.0 - frac), training_only_items
        )
    total_len = len(items) + len(training_only_items)

    training_only_prop = len(training_only_items) / total_len
    if training_only_prop >= proportions[0]:
        warnings.warn(f"training set will contain *only* training_only_corpora")
    adjusted_proportions = (
        max(proportions[0] - training_only_prop, 0),
    ) + proportions[1:]
    adjusted_proportions = tuple(
        prop / sum(adjusted_proportions) for prop in adjusted_proportions
    )
    train_items, valid_items, test_items = partition(
        adjusted_proportions, items
    )
    train_items.extend(training_only_items)
    return train_items, valid_items, test_items


def init_dirs(output_folder):
    dirname = os.path.join(output_folder, "data")
    os.makedirs(dirname, exist_ok=True)


def item_iterator(
    items: t.List[CorpusItem], verbose: bool
) -> t.Iterator[CorpusItem]:
    for i, item in enumerate(items):
        if verbose:
            print(f"{i + 1}/{len(items)}", item.csv_path)
        yield item


def segment_iter(
    df: pd.DataFrame,
    window_len: t.Optional[int],
    hop: t.Optional[int],
    window_len_jitter: t.Optional[int] = None,
    hop_jitter: t.Optional[int] = None,
    min_window_len: t.Optional[int] = None,
) -> t.Iterator[pd.DataFrame]:

    if min_window_len is None:
        min_window_len = window_len / 2
    else:
        assert min_window_len <= window_len
    if window_len_jitter is None:
        this_window_len = window_len
    else:
        window_len_l_bound = max(window_len - window_len_jitter, min_window_len)
        window_len_u_bound = window_len + window_len_jitter + 1
    if hop_jitter is None:
        this_hop = hop
    else:
        hop_l_bound = max(1, hop - hop_jitter)
        hop_u_bound = hop + hop_jitter + 1
    start_i = 0
    while start_i < len(df) - min_window_len:
        if window_len_jitter is not None:
            this_window_len = random.randint(
                window_len_l_bound, window_len_u_bound
            )
        if hop_jitter is not None:
            this_hop = random.randint(hop_l_bound, hop_u_bound)
        end_i = start_i + this_window_len
        yield df.iloc[start_i:end_i]
        start_i += this_hop


def write_symbols(writer, *symbols):
    writer.writerow(symbols)


def get_df_attrs(df, csv_path):
    score_name = os.path.basename(csv_path)
    transpose = df.attrs.get("chromatic_transpose", 0)
    scaled_by = df.attrs.get("rhythms_scaled_by", 1.0)
    return score_name, transpose, scaled_by


class CSVChunkWriter:
    def __init__(self, path_format_str, header, n_lines_per_chunk=50000):
        self._header = header
        self._fmt_str = path_format_str
        self._line_count = 0
        self._chunk_count = 0
        self._writer = None
        self._modulo = n_lines_per_chunk - 1
        self._outf = None

    def writerow(self, row: t.List[str]):
        if self._writer is None or (not self._line_count % self._modulo):
            if self._outf is not None:
                self._outf.close()
            self._chunk_count += 1
            path = self._fmt_str.format(self._chunk_count)
            self._outf = open(path, "w", newline="")
            self._writer = csv.writer(self._outf, delimiter=",")
            self._writer.writerow(self._header)
        self._writer.writerow(row)
        self._line_count += 1

    def close(self):
        if self._outf is not None:
            self._outf.close()


def write_data(
    output_folder: str,
    items: t.List[CorpusItem],
    split: str,
    ct_settings: ChordTonesDataSettings,
    repr_settings: MidiLikeSettings,
    verbose: bool = True,
):
    data_dir = get_split_dir(output_folder, split)
    format_path = os.path.join(data_dir, "{}.csv")
    os.makedirs(os.path.dirname(format_path), exist_ok=True)
    csv_chunk_writer = CSVChunkWriter(
        format_path,
        [
            "score_name",
            "transpose",
            "scaled_by",
            "start_offset",
            "events",
            "chord_tone",
        ],
    )
    try:
        init_dirs(output_folder)
        for item in item_iterator(items, verbose):
            labeled_df = pd.read_csv(item.csv_path)
            # I could augment before or after segmenting df... not entirely sure
            #   which is better. For now I'm putting augmentation first because
            #   then if we use a hop size < window_len we only need to augment
            #   each note once, rather than many times.
            for augmented_df in augment(
                split, labeled_df, ct_settings, item.synthetic
            ):
                encoded = midilike_encode(
                    augmented_df, repr_settings, feature_names=("chord_tone",)
                )
                score_name, transpose, scaled_by = get_df_attrs(
                    augmented_df, item.csv_path
                )
                for i, segment in enumerate(
                    encoded.segment(ct_settings.window_len, ct_settings.hop)
                ):
                    print("-\|/"[i % 4], end="\r", flush=True)
                    write_symbols(
                        csv_chunk_writer,
                        score_name,
                        transpose,
                        scaled_by,
                        segment["segment_onset"],
                        " ".join(segment["input"]),
                        " ".join(segment["chord_tone"]),
                    )
    finally:
        csv_chunk_writer.close()


def write_vocab(
    src_data_dir: str,
    repr_settings: MidiLikeSettings,
    inputs_vocab_path: str,
    targets_vocab_path: str,
):
    inputs_vocab = inputs_vocab_items(repr_settings)
    with open(inputs_vocab_path.format(ext="pickle"), "wb") as outf:
        pickle.dump(inputs_vocab, outf)
    with open(inputs_vocab_path.format(ext="json"), "w") as outf:
        json.dump(inputs_vocab, outf)
    with open(os.path.join(src_data_dir, "chord_tones_vocab.json")) as inf:
        targets_vocab = json.load(inf)
    with open(targets_vocab_path.format(ext="pickle"), "wb") as outf:
        pickle.dump(targets_vocab, outf)
    with open(targets_vocab_path.format(ext="json"), "w") as outf:
        json.dump(targets_vocab, outf)


def write_datasets_sub(
    src_data_dir: str,
    ct_settings: ChordTonesDataSettings,
    repr_settings: MidiLikeSettings,
    splits_todo: t.Dict[str, bool],
    output_folder: str,
    ratios: t.Tuple[Number, Number, Number] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
    vocab_only: bool = False,
):
    items_tup = get_items(
        src_data_dir=src_data_dir,
        ct_settings=ct_settings,
        repr_settings=repr_settings,
        proportions=ratios,
        frac=frac,
    )
    wrote_vocab = False
    for items, (split, todo) in zip(items_tup, splits_todo.items()):
        if todo:
            if not wrote_vocab:
                write_vocab(
                    src_data_dir, repr_settings, *vocab_paths(output_folder)
                )
                wrote_vocab = True
            if not vocab_only:
                write_data(
                    output_folder,
                    items,
                    split,
                    ct_settings,
                    repr_settings,
                )


def check_if_splits_exist(
    output_folder: str, overwrite: bool
) -> t.Dict[str, bool]:
    out = {}
    for split in ("train", "valid", "test"):
        data_path = get_split_dir(output_folder, split)
        out[split] = overwrite or not os.path.exists(data_path)
    return out


def write_datasets(
    src_data_dir: str,
    repr_args: t.Dict,
    data_args: t.Dict,
    overwrite: bool,
    frac: float = 1.0,
    ratios: t.Tuple[Number, Number, Number] = (0.8, 0.1, 0.1),
    path_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
):
    if path_kwargs is None:
        path_kwargs = {}
    ct_settings = ChordTonesDataSettings(**data_args)
    repr_settings = MidiLikeSettings(**repr_args)
    output_folder = path_from_dataclass(ct_settings)
    output_folder = path_from_dataclass(
        repr_settings,
        base_dir=output_folder,
        ratios=ratios,
        frac=frac,
        **path_kwargs,
    )
    print("Chord tones data folder: ", output_folder)
    save_dclass(ct_settings, output_folder)
    save_dclass(repr_settings, output_folder)
    splits_todo = check_if_splits_exist(output_folder, overwrite)
    if any(splits_todo.values()):
        write_datasets_sub(
            src_data_dir=src_data_dir,
            ct_settings=ct_settings,
            repr_settings=repr_settings,
            splits_todo=splits_todo,
            output_folder=output_folder,
            ratios=ratios,
            frac=frac,
        )
    else:
        print("All data exists")
    return output_folder
