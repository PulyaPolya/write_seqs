import csv
import hashlib
import json
import logging
import math
import os
import pickle
import random
import typing as t
from fractions import Fraction
from functools import cached_property
from pathlib import Path

import pandas as pd
import yaml
from music_df.add_feature import concatenate_features
from reprs import ReprEncodeError

import multiprocessing

try:
    from reprs.midi_like import MidiLikeSettings

    MIDILIKE_SUPPORTED = True
except ImportError:
    MIDILIKE_SUPPORTED = False
from reprs.oct import OctupleEncodingSettings
from reprs.shared import ReprSettingsBase

from write_seqs.utils.read_config import read_config_oc
from write_seqs.augmentations import augment
from write_seqs.settings import SequenceDataSettings, get_dataset_base_dir, save_dclass
from write_seqs.splits_utils import get_paths

import unicodedata

LOGGER = logging.getLogger(__name__)


def fraction_to_float(x):
    if not x:
        return float("nan")
    if "/" in x:
        # Convert fraction to float
        return float(Fraction(x))

    # Handle the case for integers or other numerical strings
    return float(x)


class CorpusItem:
    def __init__(
        self, csv_path: str, corpus_name: str | None = None, drop_spelling: bool = False
    ):
        if corpus_name is None:
            corpus_name = os.path.basename(os.path.dirname(csv_path))
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
        self.corpus_name = corpus_name
        self._drop_spelling = drop_spelling

    def read_df(self):
        labeled_df = pd.read_csv(
            self.csv_path,
            converters={"onset": fraction_to_float, "release": fraction_to_float},
        )
        if "Unnamed: 0" in labeled_df.columns:
            labeled_df = labeled_df.set_index("Unnamed: 0")
            labeled_df.index.name = None
        if self._drop_spelling and "spelling" in labeled_df.columns:
            labeled_df = labeled_df.drop("spelling", axis=1)

        labeled_df.attrs |= self.attrs
        labeled_df.attrs["global_key"] = self.attrs.get("global_key", None)
        labeled_df.attrs["global_key_sig"] = self.attrs.get("global_key_sig", None)
        labeled_df.attrs["pc_columns"] = self.attrs.get("pc_columns", ())
        labeled_df.attrs["pitch_columns"] = self.attrs.get("pitch_columns", ("pitch",))
        labeled_df.attrs["spelled_columns"] = self.attrs.get("spelled_columns", ())
        for col in labeled_df.attrs["spelled_columns"]:
            # We want to make sure that we're using the "b" for flat spelling
            #   style rather than "-" (so that transposition works correctly.)
            assert (
                (labeled_df[col].str.find("-") == -1) | (labeled_df[col].isna())
            ).all()
        return labeled_df

    @cached_property
    def file_size(self):
        return os.path.getsize(self.csv_path)

    @cached_property
    def int_hash(self):
        """

        >>> corpus_item = CorpusItem("/foo/bar/fake_path.csv")
        >>> corpus_item.int_hash
        126421193881301213808030691522801815677
        >>> same_corpus_item = CorpusItem("/foo/bar/fake_path.csv")
        >>> same_corpus_item.int_hash
        126421193881301213808030691522801815677
        >>> diff_corpus_item = CorpusItem("/foo/bar/other_path.csv")
        >>> diff_corpus_item.int_hash
        78042759248307369719732645949625633696
        """
        # This allows us to make certain behavior deterministic
        return int(hashlib.md5(self.csv_path.encode()).hexdigest(), 16)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.csv_path})"


def get_data_dir(output_folder: str):
    return os.path.join(output_folder, "data")


def get_split_dir(output_folder: str, split: str) -> str:
    input_path = os.path.join(get_data_dir(output_folder), split)
    return input_path


def get_items(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    proportions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
) -> tuple[list[CorpusItem], list[CorpusItem], list[CorpusItem]]:
    paths = get_paths(
        src_data_dir=src_data_dir,
        seq_settings=seq_settings,
        proportions=proportions,
        frac=frac,
    )
    items = []
    for split in paths:
        items.append(
            [CorpusItem(p, drop_spelling=seq_settings.drop_spelling) for p in split]
        )
    return tuple(items)


def init_dirs(output_folder):
    dirname = os.path.join(output_folder, "data")
    os.makedirs(dirname, exist_ok=True)


def item_iterator(
    items: list[CorpusItem], verbose: bool, start_i: int = 0, total_i: int | None = None
) -> t.Iterator[CorpusItem]:
    if total_i is None:
        total_i = len(items)
    if verbose:
        for i, item in enumerate(items, start=start_i):
            print(f"{i + 1}/{total_i}", item.csv_path)
            yield item
    else:
        yield from items


def write_symbols(writer, *symbols):
    writer.writerow(symbols)


def get_df_attrs(df):
    if "chromatic_tranpose" in df.attrs:
        transpose = df.attrs["chromatic_transpose"]
        assert "transposed_by_n_sharps" not in df.attrs
    elif "transposed_by_n_sharps" in df.attrs:
        transpose = (df.attrs["transposed_by_n_sharps"] * 7) % 12
    else:
        transpose = 0
    scaled_by = df.attrs.get("rhythms_scaled_by", 1.0)
    return transpose, scaled_by


def get_sequence_level_features(
    df: pd.DataFrame, seq_settings: SequenceDataSettings
) -> list[str]:
    out = []
    for feature in seq_settings.sequence_level_features:
        val = df.attrs[feature]
        if isinstance(val, bool):
            val = int(val)
        out.append(str(val))
    return out


class CSVChunkWriter:
    """Writes output to CSV file, dividing into chunks of `n_lines_per_chunk`."""

    def __init__(
        self,
        path_format_str,
        header,
        n_lines_per_chunk=50000,
        shared_file_counter=None,
        lock=None,
    ):
        self._header = header
        self._fmt_str = path_format_str
        self._line_count = 0
        self._chunk_count = 0
        self._shared_file_counter = shared_file_counter
        self._writer = None
        self._modulo = n_lines_per_chunk - 1
        self._outf = None
        self._lock = lock

    def writerow(self, row: list[str]):
        if self._writer is None or (not self._line_count % self._modulo):
            if self._outf is not None:
                self._outf.close()
            if self._shared_file_counter is None:
                self._chunk_count += 1
                path = self._fmt_str.format(self._chunk_count)
            else:
                assert self._lock is not None
                with self._lock:
                    self._shared_file_counter.value += 1  # type:ignore
                    path = self._fmt_str.format(
                        self._shared_file_counter.value  # type:ignore
                    )
            self._outf = open(path, "w", newline="")
            self._writer = csv.writer(self._outf, delimiter=",")
            self._writer.writerow(self._header)
        self._writer.writerow(row)
        self._line_count += 1

    def close(self):
        if self._outf is not None:
            self._outf.close()


def get_concatenated_features(
    df: pd.DataFrame, seq_settings: SequenceDataSettings, features: list[str]
):
    for concat_feature in seq_settings.concatenated_features:
        df = concatenate_features(df, concat_feature)
    return df


def write_item(
    item: CorpusItem,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
    features: list[str],
    split: str,
    csv_chunk_writer: CSVChunkWriter,
):
    labeled_df = item.read_df()

    concat_feature_names = [
        "_".join(concat_feature)
        for concat_feature in seq_settings.concatenated_features
    ]

    features = list(features) + concat_feature_names

    if not seq_settings.use_tempi:
        # Give everything the same tempo to test the effect of not using tempi
        tempo_mask = labeled_df.type == "tempo"
        labeled_df.loc[tempo_mask, "other"] = "{'tempo': 120.0}"

    # I could augment before or after segmenting df... not entirely sure
    #   which is better. For now I'm putting augmentation first because
    #   then if we use a hop size < window_len we only need to augment
    #   each note once, rather than many times.
    try:
        for augmented_df in augment(split, labeled_df, seq_settings, item.synthetic):
            augmented_df = get_concatenated_features(
                augmented_df, seq_settings, features
            )
            encoded = repr_settings.encode_f(
                augmented_df, repr_settings, feature_names=features
            )
            # (Malcolm 2023-12-01) We use the int hash to set the start offset in a
            #   deterministic way. However, it will be the same for every augmentation.
            #   It would be nice to make a different, but deterministic, hash for
            #   every augmentation.
            assert seq_settings.window_len is not None
            start_i = 0 - item.int_hash % seq_settings.window_len

            sequence_level_features = get_sequence_level_features(
                augmented_df, seq_settings
            )

            transpose, scaled_by = get_df_attrs(augmented_df)

            for i, segment in enumerate(
                encoded.segment(
                    seq_settings.window_len, seq_settings.hop, start_i=start_i
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
                    *sequence_level_features,
                )
    except ReprEncodeError:
        LOGGER.warning(f"encoding {item.csv_path} failed, skipping")


COLUMNS = [
    "score_id",
    "score_path",
    "csv_path",
    "transpose",
    "scaled_by",
    "start_offset",
    "df_indices",
    "events",
]


def get_concatenated_feature_names(seq_settings: SequenceDataSettings):
    out = []
    for concat_feature in seq_settings.concatenated_features:
        out.append("_".join(concat_feature))
    return out


def write_data_worker(
    start_i: int,
    total_i: int,
    data_chunk: list[CorpusItem],
    shared_file_counter,
    lock,
    format_path: str,
    features: list[str],
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
    verbose: bool,
    split: str,
):
    csv_chunk_writer = CSVChunkWriter(
        format_path,
        COLUMNS
        + features
        + get_concatenated_feature_names(seq_settings)
        + list(seq_settings.sequence_level_features),
        shared_file_counter=shared_file_counter,
        lock=lock,
    )
    try:
        for item in item_iterator(data_chunk, verbose, start_i, total_i):
            write_item(
                item, seq_settings, repr_settings, features, split, csv_chunk_writer
            )

    finally:
        csv_chunk_writer.close()


def chunks(list_, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(list_), n):
        yield list_[i : i + n]


def write_data(
    output_folder: str,
    items: list[CorpusItem],
    split: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
    verbose: bool = True,
    n_workers: int = 1,
):
    if not items:
        return

    items = items.copy()

    # We shuffle in the h# We shuffle in the hope that long and short items will be more or less evenly
    #   distributed between the workers
    # Actually, if we do this, it causes the metadata to be wrong!
    # random.shuffle(items)

    if seq_settings.repr_type != "oct":
        raise NotImplementedError("I need to implement 'df_indices'")
    data_dir = get_split_dir(output_folder, split)
    format_path = os.path.join(data_dir, "{}.csv")
    os.makedirs(os.path.dirname(format_path), exist_ok=True)
    features = list(seq_settings.features)

    n_workers = max(n_workers, 1)
    chunk_size = math.ceil(len(items) / n_workers)
    item_chunks = chunks(items, chunk_size)
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    shared_file_counter = manager.Value("i", 0)

    init_dirs(output_folder)

    pool = multiprocessing.Pool(processes=n_workers)
    pool.starmap(
        write_data_worker,
        [
            (
                i * chunk_size,
                len(items),
                data_chunk,
                shared_file_counter,
                lock,
                format_path,
                features,
                seq_settings,
                repr_settings,
                verbose,
                split,
            )
            for i, data_chunk in enumerate(item_chunks)
        ],
    )
    pool.close()
    pool.join()


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


def get_existing_splits_if_possible(src_data_dir: str):
    out = []
    for split in ("train", "valid", "test"):
        split_path = os.path.join(src_data_dir, split)
        if os.path.exists(split_path):
            csv_paths = [
                os.path.join(split_path, p)
                for p in os.listdir(split_path)
                if p.endswith(".csv")
            ]
            items = [CorpusItem(csv_path) for csv_path in csv_paths]
            out.append(items)
        else:
            out.append([])
    if not any(out):
        return None
    if not all(out):
        LOGGER.warning(f"At least one split of train/valid/test does not exist")
    return out


def get_items_from_input_paths(
    src_data_dir: str, seq_settings: SequenceDataSettings, input_paths_folder: str
) -> tuple[list[CorpusItem], list[CorpusItem], list[CorpusItem]]:
    items = []
    for kind in ("train", "valid", "test"):
        file_path = os.path.join(input_paths_folder, f"{kind}_paths.txt")
        if not os.path.exists(file_path):
            LOGGER.warning(f"Can't find {file_path}, skipping {kind} split")
            items.append([])
            continue
        with open(file_path) as inf:
            split = [
                unicodedata.normalize(
                    seq_settings.unicode_normalization_form,
                    os.path.join(src_data_dir, p.strip()),
                )
                for p in inf.readlines()
            ]
        for p in split:
            assert os.path.exists(p), f"{p} does not exist"
        items.append(
            [CorpusItem(p, drop_spelling=seq_settings.drop_spelling) for p in split]
        )
    return tuple(items)


def write_datasets_sub(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    splits_todo: dict[str, bool],
    output_folder: str,
    input_paths_folder: str | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
    vocab_only: bool = False,
    n_workers: int = 1,
):
    items_tup = None
    if input_paths_folder:
        items_tup = get_items_from_input_paths(
            src_data_dir, seq_settings, input_paths_folder
        )
    else:
        if seq_settings.use_existing_splits:
            items_tup = get_existing_splits_if_possible(src_data_dir)
        if items_tup is None:
            items_tup = get_items(
                src_data_dir=src_data_dir,
                seq_settings=seq_settings,
                proportions=ratios,
                frac=frac,
            )
    # I was using this to verify that result was the same across runs:
    # for x in items_tup:
    #     print(hashlib.md5(" ".join(xx.csv_path for xx in x).encode()).hexdigest())

    wrote_vocab = False
    for items, (split, todo) in zip(items_tup, splits_todo.items()):
        if todo:
            if not wrote_vocab:
                write_vocab(
                    src_data_dir,
                    seq_settings.repr_settings,
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
                    seq_settings.repr_settings,
                    n_workers=n_workers,
                )


def check_if_splits_exist(output_folder: str, overwrite: bool) -> dict[str, bool]:
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
    *,
    src_data_dir: str,
    output_dir: str,
    cli_args: list[t.Any] | None = None,
    input_paths_folder: str | None = None,
    repr_settings_path: Path | str | None = None,
    # data_settings are required because we need to specify at least the feature
    data_settings: SequenceDataSettings | None = None,
    data_settings_path: Path | str | None = None,
    overwrite: bool = False,
    frac: float = 1.0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    n_workers: int = 1,
    path_kwargs: t.Optional[dict[str, t.Any]] = None,
):
    if path_kwargs is None:
        path_kwargs = {}
    if data_settings is not None:
        seq_settings = data_settings
        assert data_settings_path is None and not cli_args
    else:
        seq_settings = read_config_oc(
            config_cls=SequenceDataSettings,
            config_path=str(data_settings_path),
            cli_args=cli_args if cli_args else [],
        )
    if repr_settings_path is not None:
        # TODO: (Malcolm 2024-03-13) eventually merge repr settings rather
        #   than overwriting them
        LOGGER.warning(
            f"{repr_settings_path=}, ignoring any repr settings provided by cli or data_settings"
        )
        if seq_settings.repr_type == "oct":
            repr_setting_cls = OctupleEncodingSettings
        elif MIDILIKE_SUPPORTED and seq_settings.repr_type == "midilike":
            repr_setting_cls = MidiLikeSettings
        else:
            raise NotImplementedError()
        repr_settings = repr_setting_cls(**load_config_from_yaml(repr_settings_path))
        seq_settings.repr_settings = repr_settings
    output_folder = os.path.join(get_dataset_base_dir(), output_dir)

    print("Chord tones data folder: ", output_folder)
    save_dclass(seq_settings, output_folder)
    save_dclass(seq_settings.repr_settings, output_folder)

    splits_todo = check_if_splits_exist(output_folder, overwrite)
    if any(splits_todo.values()):
        write_datasets_sub(
            src_data_dir=src_data_dir,
            seq_settings=seq_settings,
            splits_todo=splits_todo,
            output_folder=output_folder,
            input_paths_folder=input_paths_folder,
            ratios=ratios,
            frac=frac,
            n_workers=n_workers,
        )
    else:
        print("All data exists")
    print("Chord tones data folder: ", output_folder)
    return output_folder
