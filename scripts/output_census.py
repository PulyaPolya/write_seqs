import os
from collections import Counter, defaultdict
from itertools import chain

import pandas as pd

from write_seqs.utils.output_census_helper import conduct_census


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = conduct_census(args.input_folder)

    # if args.output_csv is not None:
    #     df.to_csv(args.output_csv)
    print(df)


if __name__ == "__main__":
    main()
