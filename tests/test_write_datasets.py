import os
import random
from pathlib import Path

import pytest

from write_seqs.write_seqs import write_datasets

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")

CONFIG_DIR = Path(os.path.dirname((os.path.realpath(__file__)))) / "test_configs"


def test_write_datasets():
    assert SRC_DATA_DIR is not None
    random.seed(42)
    seq_settings_path = CONFIG_DIR / "chord_tones_data_settings.yaml"
    write_datasets(
        SRC_DATA_DIR, "scratch", None, seq_settings_path, overwrite=True, frac=0.005
    )
