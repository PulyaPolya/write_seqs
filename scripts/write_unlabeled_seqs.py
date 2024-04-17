import glob
import math
import os
import pdb
import random
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Manager, Pool, Value

from dacite import from_dict
from omegaconf import OmegaConf
from reprs.oct import OctupleEncodingSettings

from write_seqs.settings import SequenceDataSettings
from write_seqs.write_seqs import COLUMNS, CorpusItem, CSVChunkWriter, write_item


def read_config_oc(config_path: str | None, cli_args: list[str] | None, config_cls):
    configs = []
    assert config_path is not None or cli_args is not None
    if config_path is not None:
        configs.append(OmegaConf.load(config_path))
    if cli_args is not None:
        configs.append(OmegaConf.from_cli(cli_args))
    merged_conf = OmegaConf.merge(*configs)
    resolved = OmegaConf.to_container(merged_conf, resolve=True)
    assert isinstance(resolved, dict)
    out = from_dict(data_class=config_cls, data=resolved)  # type:ignore
    return out


def get_csv_files(folder_path):
    # Use os.path.join to construct the search pattern
    search_pattern = os.path.join(folder_path, "**", "*.csv")

    # Use glob.glob with recursive=True to get all matching files
    return glob.glob(search_pattern, recursive=True)


@dataclass
class Config:
    input_folder: str
    output_folder: str
    repr_settings: OctupleEncodingSettings = field(
        default_factory=lambda: OctupleEncodingSettings()
    )
    seq_settings: SequenceDataSettings = field(
        default_factory=lambda: SequenceDataSettings(
            features=[], repr_type="oct", hop=750, window_len=1000
        )
    )
    max_files: int | None = None
    random_files: bool = False
    seed: int = 42
    num_workers: int = 8
    regex: str | None = None
    debug: bool = False
    overwrite: bool = False


def process_csv(csv_path, config, csv_chunk_writer):
    item = CorpusItem(csv_path)
    write_item(
        item,
        config.seq_settings,
        config.repr_settings,
        features=[],
        # setting split="test" will disable any augmentation
        split="test",
        csv_chunk_writer=csv_chunk_writer,
    )


def process_chunk(csv_files, config, counter, lock):
    format_path = os.path.join(config.output_folder, "{}.csv")
    csv_chunk_writer = CSVChunkWriter(format_path, COLUMNS, shared_file_counter=counter, lock=lock)
    for csv_file in csv_files:
        process_csv(csv_file, config, csv_chunk_writer)


def pdb_hook():
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)

    sys.excepthook = custom_excepthook


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    config = read_config_oc(config_path=None, cli_args=sys.argv[1:], config_cls=Config)

    if config.debug:
        pdb_hook()
    

    assert not config.seq_settings.features
    assert config.seq_settings.repr_type == "oct"
    csv_files = get_csv_files(config.input_folder)
    if config.regex is not None:
        csv_files = [f for f in csv_files if re.search(config.regex, f)]
    random.seed(config.seed)
    if config.random_files:
        random.shuffle(csv_files)

    csv_files = csv_files[: config.max_files]

    if config.overwrite and os.path.exists(config.output_folder):
        shutil.rmtree(config.output_folder)

    os.makedirs(config.output_folder, exist_ok=False)

    if config.num_workers > 1:
        file_chunks = chunks(
            csv_files, math.ceil(len(csv_files) / max(1, config.num_workers))
        )
        manager = Manager()
        counter = manager.Value("i", 0)
        lock = manager.Lock()

        with Pool(config.num_workers) as pool:
            pool.map(
                partial(process_chunk, config=config, counter=counter, lock=lock),
                file_chunks,
            )
    else:
        process_chunk(csv_files, config=config, counter=None, lock=None)
