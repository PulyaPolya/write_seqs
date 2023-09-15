import os
import random

import pytest

from write_chord_tones_seqs.write_chord_tones_seqs import write_datasets

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")


@pytest.mark.skip(
    reason="needs to be updated to match new signature for `write_datasets()`"
)
def test_write_datasets():
    random.seed(42)
    write_datasets(SRC_DATA_DIR, {}, {}, overwrite=True, frac=0.005)
