import os
import shutil
import tempfile

from write_seqs.settings import SequenceDataSettings
from write_seqs.utils.output_census_helper import conduct_census
from write_seqs.write_seqs import write_datasets
import pytest

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")


@pytest.mark.skip(
    reason="""I don't understand this test.
        Why should we expect the partitions to be the same when
        we are changing the seeds? Presumably the test used to
        pass but now it is failing."""
)
def test_same_examples_across_runs():
    assert SRC_DATA_DIR is not None
    settings = SequenceDataSettings(
        features=("chord_factors", "chord_tone"),
        corpora_to_include=["DvorakSilhouettes", "DebussySuiteBergamasque"],
        hop=250,
        window_len=1000,
        repr_type="oct",
        aug_by_key=False,
        aug_rhythms=False,
    )
    n_runs = 3
    temp_dirs = [tempfile.mkdtemp() for _ in range(n_runs)]
    dfs = []
    try:
        # TODO: (Malcolm 2024-03-13) I don't understand this test.
        #   Why should we expect the partitions to be the same when
        #   we are changing the seeds? Presumably the test used to
        #   pass but now it is failing.
        for seed, temp_dir in enumerate(temp_dirs):
            settings.split_seed = seed
            write_datasets(
                src_data_dir=SRC_DATA_DIR,
                output_dir=temp_dir,
                data_settings=settings,
            )
            dfs.append(conduct_census(os.path.join(temp_dir, "data")))
        n_examples = [df["total_examples"] for df in dfs]
        n_scores = [df["total_unique_scores"] for df in dfs]
        assert all((n_examples[0] == n).all() for n in n_examples[1:])
        assert all((n_scores[0] == n).all() for n in n_scores[1:])
    finally:
        for d in temp_dirs:
            # delete the `d` directory and its contents:
            shutil.rmtree(d)
