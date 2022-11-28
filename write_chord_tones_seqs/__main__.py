import argparse
import ast
import json
import os
import random
import sys
import warnings
from write_chord_tones_seqs.write_chord_tones_seqs import write_datasets


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore", message=r"removing grace notes")
#     warnings.filterwarnings("ignore", message=r"dangling tie at .*")
#     warnings.filterwarnings(
#         "ignore", message=r"Release of note at .* < onset of note at .*"
#     )
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--overwrite", action="store_true")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--frac", type=float, default=1.0)
#     parser.add_argument("--vocab-only", action="store_true")
#     parser.add_argument("--split-chords", action="store_true")
#     parser.add_argument("--salami-slice", action="store_true")
#     parser.add_argument("--include-barlines", action="store_true")
#     parser.add_argument("--include-corpora", nargs="+", default=())
#     parser.add_argument("--exclude-synthetic-data", action="store_true")
#     parser.add_argument("--exclude-ren-diss", action="store_true")
#     parser.add_argument("--dataset-name", type=str, default=None)
#     parser.add_argument("--n-rhythm-augs", type=int, default=2)
#     parser.add_argument("--n-trans-augs", type=int, default=12)
#     args = parser.parse_args()
#     output_folder = write_datasets(
#         overwrite=args.overwrite,
#         seed=args.seed,
#         frac=args.frac,
#         vocab_only=args.vocab_only,
#         split_chords=args.split_chords,
#         salami_slice=args.salami_slice,
#         include_barlines=args.include_barlines,
#         exclude_synthetic_data=args.exclude_synthetic_data,
#         exclude_ren_diss=args.exclude_ren_diss,
#         include_corpora=args.include_corpora,
#         dataset_name=args.dataset_name,
#         n_rhythm_augs=args.n_rhythm_augs,
#         n_trans_augs=args.n_trans_augs,
#     )
#     print(f"Data written to {output_folder}")

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")


def _sub_parse(arg_list, arg_name):
    out = {}
    if arg_list is None:
        return out
    for arg in arg_list:
        key, vals = arg.split("=", maxsplit=1)
        vals = vals.split(",")
        for i in range(len(vals)):
            try:
                vals[i] = ast.literal_eval(vals[i])
            except ValueError:
                pass
        if len(vals) == 1:
            out[key] = vals[0]
        else:
            out[key] = vals
    print(f"{arg_name} parsed as: ")
    print(json.dumps(out, indent=2))
    return out


DEFAULT_MIN_WEIGHT = 0

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r"removing grace notes")
    warnings.filterwarnings("ignore", message=r"dangling tie at .*")
    warnings.filterwarnings(
        "ignore", message=r"Release of note at .* < onset of note at .*"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr-args", nargs="*")
    parser.add_argument("--data-args", nargs="*")
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Don't include metric weights in representation.",
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=DEFAULT_MIN_WEIGHT,
        help="Minimum weight to include (only has an effect if '--no-weights' "
        f"is not passed. Default: {DEFAULT_MIN_WEIGHT}",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument(
        "--src-data-dir",
        type=str,
        default=SRC_DATA_DIR,
        help="taken from 'SRC_DATA_DIR' environment variable if not passed",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    repr_args = _sub_parse(args.repr_args, "--repr-args")
    if (
        "include_metric_weights" in repr_args
        or "min_weight_to_encode" in repr_args
    ):
        raise ValueError("Use '--no-weights' or '--min-weight' instead.")
    if "for_token_classification" in repr_args:
        raise ValueError(
            "for_token_classification isn't a valid argument (it's always true)"
        )
    repr_args["include_metric_weights"] = not args.no_weights
    repr_args["min_weight_to_encode"] = args.min_weight
    repr_args["for_token_classification"] = True
    output_folder = write_datasets(
        args.src_data_dir,
        repr_args,
        _sub_parse(args.data_args, "--data-args"),
        args.overwrite,
        args.frac,
        path_kwargs={"seed": args.seed},
    )
