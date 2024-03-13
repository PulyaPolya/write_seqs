import argparse
import os
from write_seqs.utils.read_config import read_config_oc
from write_seqs.settings import SequenceDataSettings
from write_seqs.splits_utils import get_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--src-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--proportions", nargs=3, default=[0.8, 0.1, 0.1], type=float)
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():

    args, remaining = parse_args()
    settings = read_config_oc(
        config_cls=SequenceDataSettings,
        config_path=args.config_file,
        cli_args=remaining,
    )
    paths = get_paths(args.src_data_dir, settings, args.proportions)
    os.makedirs(args.output_dir, exist_ok=True)
    for these_paths, kind in zip(paths, ("train", "valid", "test")):
        output_path = os.path.join(args.output_dir, f"{kind}_paths.txt")
        with open(output_path, "w") as outf:
            for path in these_paths:
                outf.write(f"{os.path.relpath(path, args.src_data_dir)}\n")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
