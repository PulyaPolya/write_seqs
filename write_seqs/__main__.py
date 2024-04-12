import argparse
import ast
import json
import os
import random
import sys
import warnings

from write_seqs.write_seqs import write_datasets

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")

DEFAULT_MIN_WEIGHT = 0

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r"removing grace notes")
    warnings.filterwarnings("ignore", message=r"dangling tie at .*")
    warnings.filterwarnings(
        "ignore", message=r"Release of note at .* < onset of note at .*"
    )
    parser = argparse.ArgumentParser()

    parser.add_argument("--repr-settings", type=str, help="Path to YAML file")
    parser.add_argument("--data-settings", type=str, help="Path to YAML file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="If a relative path, relative to WRITE_SEQS_BASE_DIR environment variable.",
    )
    parser.add_argument(
        "--src-data-dir",
        type=str,
        default=SRC_DATA_DIR,
        help="taken from 'SRC_DATA_DIR' environment variable if not passed",
    )
    parser.add_argument("--input-paths-dir", type=str, default=None)
    parser.add_argument(
        "--num-workers",
        default=None,
        type=int,
        help="If None, set with `os.cpu_count()`",
    )
    parser.add_argument("--msdebug", action="store_true")

    args, remaining = parser.parse_known_args()
    if args.msdebug:
        import pdb
        import traceback

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook
    random.seed(args.seed)
    output_folder = write_datasets(
        src_data_dir=args.src_data_dir,
        output_dir=args.output_dir,
        input_paths_folder=args.input_paths_dir,
        cli_args=remaining,
        repr_settings_path=args.repr_settings,
        data_settings_path=args.data_settings,
        overwrite=args.overwrite,
        frac=args.frac,
        n_workers=os.cpu_count() if args.num_workers is None else args.num_workers,
        path_kwargs={"seed": args.seed},
    )
