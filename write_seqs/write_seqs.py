import csv
import json
import logging
import os
import pickle
import random
import typing as t
from dataclasses import dataclass
from numbers import Number
from pathlib import Path

import pandas as pd
import yaml  # type:ignore
from reprs.midi_like import MidiLikeSettings  # type:ignore
from reprs.oct import OctupleEncodingSettings  # type:ignore
from reprs.shared import ReprSettingsBase  # type:ignore

from write_seqs.augmentations import augment
from write_seqs.settings import SequenceDataSettings, get_dataset_base_dir, save_dclass
from write_seqs.utils.partition import partition

LOGGER = logging.getLogger(__name__)


class CorpusItem:
    def __init__(self, csv_path):
        json_path = csv_path[:-3] + "json"
        try:
            with open(json_path) as inf:
                attrs = json.load(inf)
        except FileNotFoundError:
            attrs = {}
        self.csv_path = csv_path
        self.synthetic = attrs.get("synthetic", False)
        score_path = attrs.get("paths", csv_path)
        if not isinstance(score_path, str):
            # Assume score_path is a sequence of strings
            score_path = score_path[0]
        self.score_path = score_path
        self.score_id = attrs.get("score_name", csv_path)
        self.attrs = attrs

    def read_df(self):
        labeled_df = pd.read_csv(self.csv_path)
        labeled_df.attrs["global_key"] = self.attrs.get("global_key", None)
        labeled_df.attrs["global_key_sig"] = self.attrs.get("global_key_sig", None)
        labeled_df.attrs["pc_columns"] = self.attrs.get("pc_columns", ())
        labeled_df.attrs["pitch_columns"] = self.attrs.get("pitch_columns", ("pitch",))
        labeled_df.attrs["spelled_columns"] = self.attrs.get("spelled_columns", ())
        return labeled_df

    # csv_path: str
    # synthetic: bool = False


def get_data_dir(output_folder: str):
    return os.path.join(output_folder, "data")


def get_split_dir(output_folder: str, split: str) -> str:
    input_path = os.path.join(get_data_dir(output_folder), split)
    return input_path


def get_items_from_corpora(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
) -> t.Tuple[t.List[str], t.List[str]]:
    """Returns a pair of lists to paths, `items` and `training_only_items`.

    Called by `get_items()` below which returns train/valid/test splits.
    """
    # `corpora` are names of subfolders within the main data dir, such as `RenDissData`,
    #   `ABCData`, etc.
    _, corpora, _ = next(os.walk(src_data_dir))
    for corpus_name in seq_settings.corpora_to_exclude:
        if corpus_name not in corpora:
            LOGGER.warn(
                f"corpus '{corpus_name}' in `corpora_to_exclude` not recognized. "
                f"Valid corpora include {corpora}"
            )
    for corpus_name in seq_settings.training_only_corpora:
        if corpus_name not in corpora:
            LOGGER.warn(
                f"corpus '{corpus_name}' in `training_only_corpora` not recognized. "
                f"Valid corpora include {corpora}"
            )
    training_only_items = []
    items = []

    for corpus_name in corpora:
        print(corpus_name)
        if (
            seq_settings.corpora_to_include
            and corpus_name not in seq_settings.corpora_to_include
        ):
            continue
        if corpus_name in seq_settings.corpora_to_exclude:
            continue
        if (
            corpus_name in seq_settings.training_only_corpora
            and corpus_name not in seq_settings.corpora_to_include
        ):
            to_extend = training_only_items
        else:
            to_extend = items
        corpus_dir = os.path.join(src_data_dir, corpus_name)
        try:
            with open(os.path.join(corpus_dir, "attrs.json")) as inf:
                corpus_attrs = json.load(inf)
        except FileNotFoundError:
            corpus_attrs = {}

        if not repr_settings.validate_corpus(corpus_attrs, corpus_name):
            LOGGER.warning(
                f"Corpus {corpus_name} was not validated by {repr_settings.__class__.__name__}, skipping it"
            )
            continue
        csv_paths = [
            os.path.join(corpus_dir, p)
            for p in os.listdir(corpus_dir)
            if p.endswith(".csv")
        ]
        if (
            prop := seq_settings.corpora_sample_proportions.get(corpus_name, None)
            is not None
        ):
            csv_paths = random.sample(csv_paths, int(prop * len(csv_paths)))
        to_extend.extend([CorpusItem(csv_path) for csv_path in csv_paths])
    return items, training_only_items


def get_items(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
    seed: t.Optional[int] = None,
    proportions: t.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
    # corpora_to_exclude: t.Sequence[str] = (),
    # training_only_corpora: t.Sequence[str] = "synthetic",
) -> t.Tuple[t.List[CorpusItem], t.List[CorpusItem], t.List[CorpusItem]]:
    """Returns lists of paths for files in train, valid, and test splits, respectively."""
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
        src_data_dir, seq_settings, repr_settings
    )
    if len(items) * frac < 1:
        raise ValueError(f"{len(items)=} * {frac=} < 1")
    if frac != 1.0:
        items, _ = partition((frac, 1.0 - frac), items)
        training_only_items, _ = partition((frac, 1.0 - frac), training_only_items)
    total_len = len(items) + len(training_only_items)

    training_only_prop = len(training_only_items) / total_len
    if training_only_prop >= proportions[0]:
        LOGGER.warning(f"training set will contain *only* training_only_corpora")
    adjusted_proportions = (max(proportions[0] - training_only_prop, 0),) + proportions[
        1:
    ]
    adjusted_proportions = tuple(
        prop / sum(adjusted_proportions) for prop in adjusted_proportions
    )
    train_items, valid_items, test_items = partition(adjusted_proportions, items)
    train_items.extend(training_only_items)
    return train_items, valid_items, test_items


def init_dirs(output_folder):
    dirname = os.path.join(output_folder, "data")
    os.makedirs(dirname, exist_ok=True)


def item_iterator(items: t.List[CorpusItem], verbose: bool) -> t.Iterator[CorpusItem]:
    if verbose:
        for i, item in enumerate(items):
            print(f"{i + 1}/{len(items)}", item.csv_path)
            yield item
    else:
        yield from items


# # TODO: (Malcolm 2023-09-14) I think this function may be unused, in which case I should
# #   remove it
# def segment_iter(
#     df: pd.DataFrame,
#     window_len: int,
#     hop: t.Optional[int],
#     window_len_jitter: t.Optional[int] = None,
#     hop_jitter: t.Optional[int] = None,
#     min_window_len: t.Optional[int] = None,
# ) -> t.Iterator[pd.DataFrame]:
#     """
#     Segments data frames, optionally applying jitter.
#     """
#     if min_window_len is None:
#         min_window_len = window_len // 2
#     else:
#         assert min_window_len <= window_len
#     if window_len_jitter is None:
#         this_window_len = window_len
#     else:
#         window_len_l_bound = max(window_len - window_len_jitter, min_window_len)
#         window_len_u_bound = window_len + window_len_jitter + 1
#     if hop_jitter is None:
#         this_hop = hop
#     else:
#         assert hop is not None
#         hop_l_bound = max(1, hop - hop_jitter)
#         hop_u_bound = hop + hop_jitter + 1
#     start_i = 0
#     while start_i < len(df) - min_window_len:
#         if window_len_jitter is not None:
#             this_window_len = random.randint(
#                 window_len_l_bound, window_len_u_bound  # type:ignore
#             )
#         if hop_jitter is not None:
#             this_hop = random.randint(hop_l_bound, hop_u_bound)  # type:ignore
#         end_i = start_i + this_window_len  # type:ignore
#         yield df.iloc[start_i:end_i]
#         start_i += this_hop  # type:ignore


def write_symbols(writer, *symbols):
    writer.writerow(symbols)


def get_df_attrs(df):
    transpose = df.attrs.get("chromatic_transpose", 0)
    scaled_by = df.attrs.get("rhythms_scaled_by", 1.0)
    return transpose, scaled_by


class CSVChunkWriter:
    """Writes output to CSV file, dividing into chunks of `n_lines_per_chunk`."""

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
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
    verbose: bool = True,
):
    if seq_settings.repr_type != "oct":
        raise NotImplementedError("I need to implement 'df_indices'")
    data_dir = get_split_dir(output_folder, split)
    format_path = os.path.join(data_dir, "{}.csv")
    os.makedirs(os.path.dirname(format_path), exist_ok=True)
    features = list(seq_settings.features)
    csv_chunk_writer = CSVChunkWriter(
        format_path,
        [
            "score_id",
            "score_path",
            "csv_path",
            "transpose",
            "scaled_by",
            "start_offset",
            "df_indices",
            "events",
        ]
        + features,
    )
    try:
        init_dirs(output_folder)
        for item in item_iterator(items, verbose):
            labeled_df = item.read_df()
            # I could augment before or after segmenting df... not entirely sure
            #   which is better. For now I'm putting augmentation first because
            #   then if we use a hop size < window_len we only need to augment
            #   each note once, rather than many times.
            for augmented_df in augment(
                split, labeled_df, seq_settings, item.synthetic
            ):
                encoded = repr_settings.encode_f(
                    augmented_df, repr_settings, feature_names=features
                )

                transpose, scaled_by = get_df_attrs(augmented_df)
                for i, segment in enumerate(
                    encoded.segment(
                        seq_settings.window_len, seq_settings.hop  # type:ignore
                    )
                ):
                    feature_segments = [
                        " ".join(str(x) for x in segment[f]) for f in features
                    ]
                    print("-\\|/"[i % 4], end="\r", flush=True)
                    write_symbols(
                        csv_chunk_writer,
                        item.score_id,
                        item.score_path,
                        item.csv_path,
                        transpose,
                        scaled_by,
                        segment["segment_onset"],  # type:ignore
                        segment["df_indices"],
                        " ".join(segment["input"]),  # type:ignore
                        *feature_segments,
                    )
    finally:
        csv_chunk_writer.close()


def write_vocab(
    src_data_dir: str,
    repr_settings: ReprSettingsBase,
    output_folder: str,
    features: t.Iterable[str],
):
    inputs_vocab_path = os.path.join(output_folder, "inputs_vocab.list.{ext}")
    targets_vocab_path = os.path.join(
        output_folder, "targets_{feature_i}_vocab.list.{ext}"
    )
    feature_names_path = os.path.join(output_folder, "feature_names.json")

    inputs_vocab = repr_settings.inputs_vocab
    with open(inputs_vocab_path.format(ext="pickle"), "wb") as outf:
        pickle.dump(inputs_vocab, outf)
    with open(inputs_vocab_path.format(ext="json"), "w") as outf:
        json.dump(inputs_vocab, outf)
    with open(feature_names_path, "w") as outf:
        json.dump(list(features), outf)

    missing_vocabs = []
    for feature_i, feature in enumerate(features):
        try:
            with open(os.path.join(src_data_dir, f"{feature}_vocab.json")) as inf:
                targets_vocab = json.load(inf)
        except FileNotFoundError:
            try:
                # For backwards compatibility, add plural "s"
                with open(os.path.join(src_data_dir, f"{feature}s_vocab.json")) as inf:
                    targets_vocab = json.load(inf)
            except FileNotFoundError:
                missing_vocabs.append((feature_i, feature))
            continue
        with open(
            targets_vocab_path.format(feature_i=feature_i, ext="pickle"), "wb"
        ) as outf:
            pickle.dump(targets_vocab, outf)
        with open(
            targets_vocab_path.format(feature_i=feature_i, ext="json"), "w"
        ) as outf:
            json.dump(targets_vocab, outf)
    if missing_vocabs:
        for feature_i, feature in missing_vocabs:
            LOGGER.warning(f"Missing vocab file for {feature_i=}, {feature}")


def write_datasets_sub(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
    splits_todo: t.Dict[str, bool],
    output_folder: str,
    ratios: t.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
    vocab_only: bool = False,
):
    items_tup = get_items(
        src_data_dir=src_data_dir,
        seq_settings=seq_settings,
        repr_settings=repr_settings,
        proportions=ratios,
        frac=frac,
    )
    wrote_vocab = False
    for items, (split, todo) in zip(items_tup, splits_todo.items()):
        if todo:
            if not wrote_vocab:
                write_vocab(
                    src_data_dir,
                    repr_settings,
                    output_folder,
                    seq_settings.features,
                )
                wrote_vocab = True
            if not vocab_only:
                write_data(
                    output_folder,
                    items,
                    split,
                    seq_settings,
                    repr_settings,
                )


def check_if_splits_exist(output_folder: str, overwrite: bool) -> t.Dict[str, bool]:
    out = {}
    for split in ("train", "valid", "test"):
        data_path = get_split_dir(output_folder, split)
        out[split] = overwrite or not os.path.exists(data_path)
    return out


def load_config_from_yaml(yaml_path: Path | str | None) -> dict:
    if yaml_path is None:
        return {}
    with open(yaml_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def write_datasets(
    src_data_dir: str,
    output_dir: str,
    # repr_type: t.Literal["oct", "midilike"],
    repr_settings: Path | str | None,
    # data_settings are required because we need to specify at least the feature
    data_settings: Path | str,
    overwrite: bool,
    frac: float = 1.0,
    ratios: t.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    path_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
):
    if path_kwargs is None:
        path_kwargs = {}
    seq_settings = SequenceDataSettings(**load_config_from_yaml(data_settings))
    if seq_settings.repr_type == "oct":
        repr_setting_cls = OctupleEncodingSettings
    elif seq_settings.repr_type == "midilike":
        repr_setting_cls = MidiLikeSettings
    else:
        raise NotImplementedError()
    repr_settings_inst = repr_setting_cls(**load_config_from_yaml(repr_settings))
    output_folder = os.path.join(get_dataset_base_dir(), output_dir)
    # output_folder = path_from_dataclass(seq_settings)
    # output_folder = path_from_dataclass(
    #     repr_settings,
    #     base_dir=output_folder,
    #     ratios=ratios,
    #     frac=frac,
    #     **path_kwargs,
    # )
    print("Chord tones data folder: ", output_folder)
    save_dclass(seq_settings, output_folder)
    save_dclass(repr_settings_inst, output_folder)
    # with open(os.path.join(output_folder, "repr.txt"), "w") as outf:
    #     outf.write(repr_type)

    splits_todo = check_if_splits_exist(output_folder, overwrite)
    if any(splits_todo.values()):
        write_datasets_sub(
            src_data_dir=src_data_dir,
            seq_settings=seq_settings,
            repr_settings=repr_settings_inst,
            splits_todo=splits_todo,
            output_folder=output_folder,
            ratios=ratios,
            frac=frac,
        )
    else:
        print("All data exists")
    print("Chord tones data folder: ", output_folder)
    return output_folder
