import json
import logging
import os
import random
from typing import Sequence, Literal
from collections import defaultdict
from reprs.shared import ReprSettingsBase

from write_seqs.settings import SequenceDataSettings
from write_seqs.utils.partition import partition

LOGGER = logging.getLogger(__name__)


def _get_paths_from_corpora(
    src_data_dir: str,
    repr_settings: ReprSettingsBase | None,
    corpora_to_exclude: Sequence[str] = (),
    training_only_corpora: Sequence[str] = (),
    corpora_to_include: Sequence[str] = (),
    synthetic_corpora_to_include: Sequence[str] = (),
    corpora_sample_proportions: None | dict[str, float] = None,
    output_format: Literal["flat_list", "dict_by_corpus"] = "flat_list",
) -> tuple[list[str], list[str]] | tuple[dict[str, list[str]], list[str]]:
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

    for corpus_name in corpora_to_exclude:
        if corpus_name not in corpora:
            LOGGER.warning(
                f"corpus '{corpus_name}' in `corpora_to_exclude` not recognized. "
                f"Valid corpora include {corpora}"
            )
    for corpus_name in training_only_corpora:
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
            corpora_to_include
            and corpus_name not in corpora_to_include
            and corpus_name not in synthetic_corpora_to_include
        ):
            continue
        if corpus_name in corpora_to_exclude:
            continue
        corpus_dir = os.path.join(src_data_dir, corpus_name)
        try:
            with open(os.path.join(corpus_dir, "attrs.json")) as inf:
                corpus_attrs = json.load(inf)
        except FileNotFoundError:
            corpus_attrs = {}

        if (
            corpus_attrs.get("synthetic")
            and corpus_name not in synthetic_corpora_to_include
        ):
            continue

        if (
            corpus_name in training_only_corpora
            and corpus_name not in corpora_to_include
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
        if corpora_sample_proportions and (
            prop := corpora_sample_proportions.get(corpus_name, None) is not None
        ):
            csv_paths = random.sample(csv_paths, int(prop * len(csv_paths)))
        to_extend.extend(
            csv_paths
            # TODO: (Malcolm 2024-03-13) do we need corpus_name and drop_spelling attrs?
            # [
            #     # CorpusItem(csv_path, corpus_name, seq_settings.drop_spelling)
            #     # for csv_path in csv_paths
            # ]
        )

    training_only_items = sorted(training_only_items)
    # We sort to be sure that the result will be stable
    if output_format == "flat_list":
        assert isinstance(items, list)
        items = sorted(items)
        return items, training_only_items

    else:
        assert isinstance(items, dict)
        for corpus_name in corpora:
            items[corpus_name] = sorted(items[corpus_name])
        return items, training_only_items


def get_paths_within_corpora(**kwargs) -> tuple[dict[str, list[str]], list[str]]:
    assert "output_format" not in kwargs
    out = _get_paths_from_corpora(**kwargs, output_format="dict_by_corpus")
    assert isinstance(out[0], dict) and isinstance(out[1], list)
    return out  # type:ignore


def get_paths_across_corpora(**kwargs) -> tuple[list[str], list[str]]:
    assert "output_format" not in kwargs
    out = _get_paths_from_corpora(**kwargs, output_format="flat_list")
    assert isinstance(out[0], list) and isinstance(out[1], list)
    return out  # type:ignore


def handle_partition(
    paths: list[str],
    training_only_paths: list[str] | None = None,
    proportions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
    proportions_exclude_training_only_paths: bool = True,
) -> tuple[list[str], list[str], list[str], list[str] | None]:
    file_sizes = [os.path.getsize(p) for p in paths]

    if training_only_paths is not None:
        training_only_file_sizes = [os.path.getsize(p) for p in training_only_paths]

    if frac < 1.0:
        # Get a random subset of all paths
        paths, _ = partition(
            (frac, 1.0 - frac), paths, file_sizes  # type:ignore
        )
        if training_only_paths is not None:
            training_only_paths, _ = partition(
                (frac, 1.0 - frac),
                training_only_paths,
                training_only_file_sizes,  # type:ignore
            )

    if proportions_exclude_training_only_paths:
        train_paths, valid_paths, test_paths = partition(
            proportions, paths, file_sizes  # type:ignore
        )
    else:
        assert training_only_paths is not None
        training_only_size = sum(training_only_file_sizes)
        total_size = sum(file_sizes) + training_only_size
        training_only_prop = training_only_size / total_size
        if training_only_prop >= proportions[0]:
            LOGGER.warning(f"training set will contain *only* training_only_corpora")
        adjusted_proportions = (
            max(proportions[0] - training_only_prop, 0),
        ) + proportions[1:]
        adjusted_proportions = tuple(
            prop / sum(adjusted_proportions) for prop in adjusted_proportions
        )
        train_paths, valid_paths, test_paths = partition(
            adjusted_proportions, paths, file_sizes  # type:ignore
        )
    return train_paths, valid_paths, test_paths, training_only_paths


def get_paths(
    src_data_dir: str,
    seq_settings: SequenceDataSettings,
    proportions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    frac: float = 1.0,
) -> tuple[list[str], list[str], list[str]]:
    """Returns lists of paths for files in train, valid, and test splits, respectively."""
    if seq_settings.split_seed is not None:
        random.seed(seq_settings.split_seed)
    if seq_settings.split_by_corpora:
        if not seq_settings.proportions_exclude_training_only_items:
            raise NotImplementedError
        paths, training_only_paths = get_paths_within_corpora(
            src_data_dir=src_data_dir,
            repr_settings=seq_settings.repr_settings,
            corpora_to_exclude=seq_settings.corpora_to_exclude,
            training_only_corpora=seq_settings.training_only_corpora,
            corpora_to_include=seq_settings.corpora_to_include,
            synthetic_corpora_to_include=seq_settings.synthetic_corpora_to_include,
            corpora_sample_proportions=seq_settings.corpora_sample_proportions,
        )
        training_only_file_sizes = [os.path.getsize(p) for p in training_only_paths]
        training_only_paths, _ = partition(
            (frac, 1.0 - frac),
            training_only_paths,
            training_only_file_sizes,  # type:ignore
        )

        len_paths = sum(len(sub_paths) for sub_paths in paths.values())
        if len_paths * frac < 1:
            raise ValueError(f"{src_data_dir=} {len_paths=} * {frac=} < 1")
        train_paths = []
        valid_paths = []
        test_paths = []

        for corpus_name in paths:
            c_train_paths, c_valid_paths, c_test_paths, _ = handle_partition(
                paths[corpus_name],
                proportions=proportions,
                frac=frac,
                proportions_exclude_training_only_paths=True,
            )
            train_paths.extend(c_train_paths)
            valid_paths.extend(c_valid_paths)
            test_paths.extend(c_test_paths)

    else:
        paths, training_only_paths = get_paths_across_corpora(
            src_data_dir=src_data_dir,
            repr_settings=seq_settings.repr_settings,
            corpora_to_exclude=seq_settings.corpora_to_exclude,
            training_only_corpora=seq_settings.training_only_corpora,
            corpora_to_include=seq_settings.corpora_to_include,
            synthetic_corpora_to_include=seq_settings.synthetic_corpora_to_include,
            corpora_sample_proportions=seq_settings.corpora_sample_proportions,
        )

        # I was using this to verify that result was the same across runs:
        # for x in (items, training_only_items):
        #     print(hashlib.md5(" ".join(xx.csv_path for xx in x).encode()).hexdigest())

        if len(paths) * frac < 1:
            raise ValueError(f"{src_data_dir=} {len(paths)=} * {frac=} < 1")
        train_paths, valid_paths, test_paths, training_only_paths = handle_partition(
            paths,
            training_only_paths=training_only_paths,
            proportions=proportions,
            frac=frac,
            proportions_exclude_training_only_paths=seq_settings.proportions_exclude_training_only_items,
        )
        assert training_only_paths is not None

    train_paths.extend(training_only_paths)
    return train_paths, valid_paths, test_paths
