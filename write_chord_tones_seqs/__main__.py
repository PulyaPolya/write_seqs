import argparse
import ast
import json
import os
import random
import sys
import warnings

from write_chord_tones_seqs.write_chord_tones_seqs import write_datasets

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")

DEFAULT_MIN_WEIGHT = 0

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r"removing grace notes")
    warnings.filterwarnings("ignore", message=r"dangling tie at .*")
    warnings.filterwarnings(
        "ignore", message=r"Release of note at .* < onset of note at .*"
    )
    parser = argparse.ArgumentParser()
    # parser.add_argument("--repr-type", choices=["midilike", "oct"], required=True)
    parser.add_argument("--repr-settings", type=str, help="Path to YAML file")
    parser.add_argument("--data-settings", type=str, help="Path to YAML file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="If a relative path, relative to CT_SEQS_BASE_DIR environment variable.",
    )
    parser.add_argument(
        "--src-data-dir",
        type=str,
        default=SRC_DATA_DIR,
        help="taken from 'SRC_DATA_DIR' environment variable if not passed",
    )
    parser.add_argument("--msdebug", action="store_true")

    args = parser.parse_args()
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
        args.src_data_dir,
        args.output_dir,
        # args.repr_type,
        args.repr_settings,
        args.data_settings,
        args.overwrite,
        args.frac,
        path_kwargs={"seed": args.seed},
    )
