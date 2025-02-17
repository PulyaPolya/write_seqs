import argparse
import ast
import json
import os
import random
import sys
import warnings

from write_seqs.write_seqs.write_seqs import write_datasets      #modified

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")

DEFAULT_MIN_WEIGHT = 0

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", message=r"removing grace notes")
    # warnings.filterwarnings("ignore", message=r"dangling tie at .*")
    # warnings.filterwarnings(
    #     "ignore", message=r"Release of note at .* < onset of note at .*"
    # )
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--repr-settings", type=str, help="Path to YAML file")
    # parser.add_argument("--data-settings", type=str, help="Path to YAML file")
    # parser.add_argument("--overwrite", action="store_true")
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--frac", type=float, default=1.0)
    # parser.add_argument(
    #     "--output-dir",
    #     required=True,
    #     help="If a relative path, relative to WRITE_SEQS_BASE_DIR environment variable.",
    # )
    # parser.add_argument(
    #     "--src-data-dir",
    #     type=str,
    #     default=SRC_DATA_DIR,
    #     help="taken from 'SRC_DATA_DIR' environment variable if not passed",
    # )
    # parser.add_argument("--input-paths-dir", type=str, default=None)
    # parser.add_argument(
    #     "--num-workers",
    #     default=None,
    #     type=int,
    #     help="If None, set with `os.cpu_count()`",
    # )
    # parser.add_argument("--msdebug", action="store_true")

    # args, remaining = parser.parse_known_args()
    # if args.msdebug:
    #     import pdb
    #     import traceback

    #     def custom_excepthook(exc_type, exc_value, exc_traceback):
    #         traceback.print_exception(
    #             exc_type, exc_value, exc_traceback, file=sys.stdout
    #         )
    #         pdb.post_mortem(exc_traceback)

    #     sys.excepthook = custom_excepthook
    # random.seed(args.seed)
    # print(args.repr_settings)
    # print(args.data_settings)
    # output_folder = write_datasets(
    #     src_data_dir=args.src_data_dir,
    #     output_dir=args.output_dir,
    #     input_paths_folder=args.input_paths_dir,
    #     cli_args=remaining,
    #     repr_settings_path=args.repr_settings,
    #     data_settings_path=args.data_settings,
    #     overwrite=args.overwrite,
    #     frac=args.frac,
    #     n_workers=os.cpu_count() if args.num_workers is None else args.num_workers,
    #     path_kwargs={"seed": args.seed},
    # )
    print("something")
    warnings.filterwarnings("ignore", message=r"removing grace notes")
    warnings.filterwarnings("ignore", message=r"dangling tie at .*")
    warnings.filterwarnings("ignore", message=r"Release of note at .* < onset of note at .*")

    # -----------------------------
    # Define Parameters Manually
    # -----------------------------

    # These are similar to your shell parameters.
    # RNBERT_DIR is defined relative to this file.
    RNBERT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Number of workers (default 16 or set as desired)
    NUM_WORKERS =16

    # ZIP_FILE may be used later (if needed)
    ZIP_FILE = os.path.join(RNBERT_DIR, "dataset.zip")

    # RNDATA_ROOT is taken from the environment if set; otherwise default to ~/datasets
    RNDATA_ROOT = os.environ.get("RNDATA_ROOT", os.path.join(os.path.expanduser("~"), "datasets"))

    # CSV_DIR: directory with CSV files (source data)
    CSV_DIR = os.environ.get("CSV_DIR", os.path.join(RNDATA_ROOT, "rnbert_csvs"))

    # SEQS_DIR: output directory for sequences
    SEQS_DIR = os.environ.get("SEQS_DIR", os.path.join(RNDATA_ROOT, "rnbert_seqs"))

    # FAIRSEQ_ABSTRACT_RAW: another directory parameter if needed (not used below)
    FAIRSEQ_ABSTRACT_RAW = os.environ.get("FAIRSEQ_ABSTRACT_RAW", os.path.join(RNDATA_ROOT, "rnbert_abstract_data_raw"))

    # Additional parameters for write_datasets:
    seed = 42
    frac = 1.0
    overwrite = True  # Set to True if you want to overwrite existing output
    msdebug = False    # Set to True if you want to enable the debug exception hook

    # Paths for YAML settings files (update these paths accordingly)
    repr_settings_path = os.path.join(RNBERT_DIR, "path", "to", "repr_settings.yaml")
    data_settings_path   = os.path.join(RNBERT_DIR, "path", "to", "data_settings.yaml")

    # For write_datasets, we need:
    #   - src_data_dir: your CSV source directory
    #   - output_dir: where to write the sequences (SEQS_DIR)
    #   - input_paths_folder: if you have a folder with input paths, else None
    #   - cli_args: extra command-line arguments (empty here)
    #   - n_workers: the number of parallel workers
    #   - path_kwargs: extra parameters to pass (here, just the seed)
    src_data_dir = CSV_DIR
    output_dir = SEQS_DIR
    input_paths_dir = None
    cli_args = []  # No additional command-line arguments in this call

    # Set the random seed for reproducibility
    random.seed(seed)

    # Optionally, set up a custom exception hook for debugging if msdebug is enabled.
    if msdebug:
        import pdb
        import traceback

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook

    # -----------------------------
    # Call write_datasets Directly
    # -----------------------------
    output_folder = write_datasets(
        src_data_dir=src_data_dir,
        output_dir=output_dir,
        input_paths_folder=input_paths_dir,
        cli_args=cli_args,
        repr_settings_path=None,
        data_settings_path="C:\\Polina\\work\\research_project\\rnbert\\write_seqs\\configs\\oct_data_abstract.yaml",
        overwrite=overwrite,
        frac=frac,
        n_workers=NUM_WORKERS,
        path_kwargs={"seed": seed},
    )

