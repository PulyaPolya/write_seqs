import csv
import hashlib
import json
import logging
import os
import pickle
import random
import typing as t
from collections import defaultdict
from fractions import Fraction
from functools import cached_property
from multiprocessing import Lock, Value
from pathlib import Path

import pandas as pd
import yaml
from reprs import ReprEncodeError
from reprs.midi_like import MidiLikeSettings
from reprs.oct import OctupleEncodingSettings
from reprs.shared import ReprSettingsBase

from write_seqs.augmentations import augment
from write_seqs.settings import SequenceDataSettings, get_dataset_base_dir, save_dclass
from write_seqs.utils.partition import partition

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

    # csv_path: str
    # synthetic: bool = False


def get_data_dir(output_folder: str):
    return os.path.join(output_folder, "data")


def get_split_dir(output_folder: str, split: str) -> str:
    input_path = os.path.join(get_data_dir(output_folder), split)
    return input_path


def _get_items_from_corpora(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase | None,
    output_format: t.Literal["flat_list", "dict_by_corpus"] = "flat_list",
) -> (
    tuple[list[CorpusItem], list[CorpusItem]]
    | tuple[dict[str, list[CorpusItem]], list[CorpusItem]]
):
    """Returns a pair of lists to paths, `items` and `training_only_items`.

    Called by `get_items()` below which returns train/valid/test splits.
    """
    # `corpora` are names of subfolders within the main data dir, such as `RenDissData`,
    #   `ABCData`, etc.
    _, corpora, _ = next(os.walk(src_data_dir))
    if not corpora:
        LOGGER.error(
            f"Found no corpora; there should be at least one subdirectory of `{src_data_dir}`"
        )
        raise ValueError(
            f"Found no corpora; there should be at least one subdirectory of `{src_data_dir}`"
        )

    for corpus_name in seq_settings.corpora_to_exclude:
        if corpus_name not in corpora:
            LOGGER.warning(
                f"corpus '{corpus_name}' in `corpora_to_exclude` not recognized. "
                f"Valid corpora include {corpora}"
            )
    for corpus_name in seq_settings.training_only_corpora:
        if corpus_name not in corpora:
            LOGGER.warning(
                f"corpus '{corpus_name}' in `training_only_corpora` not recognized. "
                f"Valid corpora include {corpora}"
            )
    training_only_items = []

    if output_format == "flat_list":
        items = []
    else:
        items = defaultdict(list)

    # We sort corpora to make output stable
    for corpus_name in sorted(corpora):
        if (
            seq_settings.corpora_to_include
            and corpus_name not in seq_settings.corpora_to_include
            and corpus_name not in seq_settings.synthetic_corpora_to_include
        ):
            continue
        if corpus_name in seq_settings.corpora_to_exclude:
            continue
        corpus_dir = os.path.join(src_data_dir, corpus_name)
        try:
            with open(os.path.join(corpus_dir, "attrs.json")) as inf:
                corpus_attrs = json.load(inf)
        except FileNotFoundError:
            corpus_attrs = {}

        if (
            corpus_attrs.get("synthetic")
            and corpus_name not in seq_settings.synthetic_corpora_to_include
        ):
            continue

        if (
            corpus_name in seq_settings.training_only_corpora
            and corpus_name not in seq_settings.corpora_to_include
        ) or corpus_attrs.get("synthetic", False):
            to_extend = training_only_items
        else:
            if output_format == "flat_list":
                assert isinstance(items, list)
                to_extend = items
            else:
                assert isinstance(items, dict)
                to_extend = items[corpus_name]

        if repr_settings is not None and not repr_settings.validate_corpus(
            corpus_attrs, corpus_name
        ):
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
        to_extend.extend(
            [
                CorpusItem(csv_path, corpus_name, seq_settings.drop_spelling)
                for csv_path in csv_paths
            ]
        )

    training_only_items = sorted(training_only_items, key=lambda x: x.csv_path)
    # We sort to be sure that the result will be stable
    if output_format == "flat_list":
        assert isinstance(items, list)
        items = sorted(items, key=lambda x: x.csv_path)
        return items, training_only_items

    else:
        assert isinstance(items, dict)
        for corpus_name in corpora:
            items[corpus_name] = sorted(items[corpus_name], key=lambda x: x.csv_path)
        return items, training_only_items


def get_items_within_corpora(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase | None,
) -> tuple[dict[str, list[CorpusItem]], list[CorpusItem]]:
    out = _get_items_from_corpora(
        src_data_dir, seq_settings, repr_settings, output_format="dict_by_corpus"
    )
    assert isinstance(out[0], dict) and isinstance(out[1], list)
    return out  # type:ignore


def get_items_across_corpora(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase | None,
) -> tuple[list[CorpusItem], list[CorpusItem]]:
    out = _get_items_from_corpora(
        src_data_dir, seq_settings, repr_settings, output_format="flat_list"
    )
    assert isinstance(out[0], list) and isinstance(out[1], list)
    return out  # type:ignore


def handle_partition(
    items: list[CorpusItem],
    training_only_items: list[CorpusItem] | None = None,
    proportions: t.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
    proportions_exclude_training_only_items: bool = True,
) -> tuple[
    list[CorpusItem], list[CorpusItem], list[CorpusItem], list[CorpusItem] | None
]:
    if frac < 1.0:
        # Get a random subset of all items
        items, _ = partition(
            (frac, 1.0 - frac), items, [item.file_size for item in items]
        )
        if training_only_items is not None:
            training_only_items, _ = partition(
                (frac, 1.0 - frac),
                training_only_items,
                [item.file_size for item in training_only_items],
            )

    if proportions_exclude_training_only_items:
        train_items, valid_items, test_items = partition(
            proportions, items, [item.file_size for item in items]
        )
    else:
        assert training_only_items is not None
        training_only_size = sum(item.file_size for item in training_only_items)
        total_size = sum(item.file_size for item in items) + training_only_size
        training_only_prop = training_only_size / total_size
        if training_only_prop >= proportions[0]:
            LOGGER.warning(f"training set will contain *only* training_only_corpora")
        adjusted_proportions = (
            max(proportions[0] - training_only_prop, 0),
        ) + proportions[1:]
        adjusted_proportions = tuple(
            prop / sum(adjusted_proportions) for prop in adjusted_proportions
        )
        train_items, valid_items, test_items = partition(
            adjusted_proportions, items, [item.file_size for item in items]
        )
    return train_items, valid_items, test_items, training_only_items


def get_items(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase | None,
    proportions: t.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
) -> t.Tuple[t.List[CorpusItem], t.List[CorpusItem], t.List[CorpusItem]]:
    """Returns lists of paths for files in train, valid, and test splits, respectively."""
    if seq_settings.split_seed is not None:
        random.seed(seq_settings.split_seed)
    if seq_settings.split_by_corpora:
        if not seq_settings.proportions_exclude_training_only_items:
            raise NotImplementedError
        items, training_only_items = get_items_within_corpora(
            src_data_dir, seq_settings, repr_settings
        )
        training_only_items, _ = partition(
            (frac, 1.0 - frac),
            training_only_items,
            [item.file_size for item in training_only_items],
        )

        len_items = sum(len(sub_items) for sub_items in items.values())
        if len_items * frac < 1:
            raise ValueError(f"{src_data_dir=} {len_items=} * {frac=} < 1")
        train_items = []
        valid_items = []
        test_items = []

        for corpus_name in items:
            c_train_items, c_valid_items, c_test_items, _ = handle_partition(
                items[corpus_name],
                proportions=proportions,
                frac=frac,
                proportions_exclude_training_only_items=True,
            )
            train_items.extend(c_train_items)
            valid_items.extend(c_valid_items)
            test_items.extend(c_test_items)

    else:
        items, training_only_items = get_items_across_corpora(
            src_data_dir, seq_settings, repr_settings
        )

        # I was using this to verify that result was the same across runs:
        # for x in (items, training_only_items):
        #     print(hashlib.md5(" ".join(xx.csv_path for xx in x).encode()).hexdigest())

        if len(items) * frac < 1:
            raise ValueError(f"{src_data_dir=} {len(items)=} * {frac=} < 1")
        train_items, valid_items, test_items, training_only_items = handle_partition(
            items,
            training_only_items=training_only_items,
            proportions=proportions,
            frac=frac,
            proportions_exclude_training_only_items=seq_settings.proportions_exclude_training_only_items,
        )
        assert training_only_items is not None

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


def write_symbols(writer, *symbols):
    writer.writerow(symbols)


def get_df_attrs(df):
    transpose = df.attrs.get("chromatic_transpose", 0)
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
    ):
        self._header = header
        self._fmt_str = path_format_str
        self._line_count = 0
        self._chunk_count = 0
        self._shared_file_counter = shared_file_counter
        self._writer = None
        self._modulo = n_lines_per_chunk - 1
        self._outf = None

    def writerow(self, row: t.List[str]):
        if self._writer is None or (not self._line_count % self._modulo):
            if self._outf is not None:
                self._outf.close()
            if self._shared_file_counter is None:
                self._chunk_count += 1
                path = self._fmt_str.format(self._chunk_count)
            else:
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


def write_item(
    item: CorpusItem,
    seq_settings: SequenceDataSettings,
    repr_settings: ReprSettingsBase,
    features: t.Sequence[str],
    split: str,
    csv_chunk_writer: CSVChunkWriter,
):
    labeled_df = item.read_df()

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
        format_path, COLUMNS + features + list(seq_settings.sequence_level_features)
    )
    try:
        init_dirs(output_folder)
        for item in item_iterator(items, verbose):
            write_item(
                item, seq_settings, repr_settings, features, split, csv_chunk_writer
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
    items_tup = None
    if seq_settings.use_existing_splits:
        items_tup = get_existing_splits_if_possible(src_data_dir)
    if items_tup is None:
        items_tup = get_items(
            src_data_dir=src_data_dir,
            seq_settings=seq_settings,
            repr_settings=repr_settings,
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
    data_settings: Path | str | SequenceDataSettings,
    overwrite: bool = False,
    frac: float = 1.0,
    ratios: t.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    path_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
):
    if path_kwargs is None:
        path_kwargs = {}
    if isinstance(data_settings, SequenceDataSettings):
        seq_settings = data_settings
    else:
        seq_settings = SequenceDataSettings(**load_config_from_yaml(data_settings))
    if seq_settings.repr_type == "oct":
        repr_setting_cls = OctupleEncodingSettings
    elif seq_settings.repr_type == "midilike":
        repr_setting_cls = MidiLikeSettings
    else:
        raise NotImplementedError()
    repr_settings_inst = repr_setting_cls(**load_config_from_yaml(repr_settings))
    output_folder = os.path.join(get_dataset_base_dir(), output_dir)

    print("Chord tones data folder: ", output_folder)
    save_dclass(seq_settings, output_folder)
    save_dclass(repr_settings_inst, output_folder)

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
