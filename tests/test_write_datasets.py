import os
import random

from write_chord_tones_seqs.write_chord_tones_seqs import write_datasets

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")


def test_write_datasets():
    random.seed(42)
    write_datasets(SRC_DATA_DIR, {}, {}, overwrite=True, frac=0.005)
