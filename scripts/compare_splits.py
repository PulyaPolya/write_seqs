import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from write_seqs.settings import SequenceDataSettings
from write_seqs.write_seqs import get_items

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-data-dir", default=SRC_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        default="/Users/malcolm/Dropbox/Yale_Stuff/dissertation/supporting_files/latex_tables",
    )
    parser.add_argument("--dont-split-by-corpora", action="store_true")
    args = parser.parse_args()
    return args


N_SEEDS = 100


def main():
    args = parse_args()
    seq_settings = SequenceDataSettings(
        features=["dummy"], split_by_corpora=not args.dont_split_by_corpora
    )
    score_results = []
    size_results = []
    for seed in range(42, 42 + N_SEEDS):
        score_counts = {}
        total_sizes = {}
        seq_settings.split_seed = seed
        splits = get_items(
            args.src_data_dir, seq_settings=seq_settings, repr_settings=None
        )
        for split, split_name in zip(splits, ("train", "valid", "test")):
            counts = Counter(c.corpus_name for c in split)
            sizes = Counter()
            for c in split:
                sizes[c.corpus_name] += c.file_size
            # counts["total"] = sum(counts.values())
            # sizes["total"] = sum(sizes.values())
            score_counts[f"{split_name}_counts_{seed}"] = counts
            total_sizes[f"{split_name}_sizes_{seed}"] = sizes

        def _get_df(d):
            df = pd.DataFrame(d)
            df = df.fillna(value=0)
            for col in df.columns:
                df[col] = df[col].astype(int)
            df.iloc[:, :] = df.values / df.values.sum(axis=1, keepdims=True)
            df = df.sort_index()
            df.loc["Total"] = df.values.sum(axis=0)
            df.loc["Mean"] = df.values[:-1].mean(axis=0)
            return df

        score_df = _get_df(score_counts)
        size_df = _get_df(total_sizes)
        score_results.append(score_df)
        size_results.append(size_df)

    # dfs = {split_name: pd.DataFrame(data) for split_name, data in score_counts.items()}
    def _aggregate(results: list[pd.DataFrame]):
        arr = np.stack([df.values for df in results], axis=0)
        means = arr.mean(axis=0)
        means_df = results[0].copy()
        means_df.loc[:, :] = means
        # Underscores cause latex output with pandoc to fail
        means_df.columns = [
            col.rsplit("_", maxsplit=1)[0].replace("_", " ").capitalize() + " mean"
            for col in means_df.columns
        ]
        stds = arr.std(axis=0)
        stds_df = results[0].copy()
        stds_df.loc[:, :] = stds
        stds_df.columns = [
            col.rsplit("_", maxsplit=1)[0].replace("_", " ").capitalize() + " std"
            for col in stds_df.columns
        ]
        return means_df, stds_df

    score_means, score_stds = _aggregate(score_results)
    size_means, size_stds = _aggregate(size_results)
    print(score_means)
    print(score_stds)
    print(size_means)
    print(size_stds)
    if args.output_dir is not None:
        info_str = "by_all" if args.dont_split_by_corpora else "by_corpus"
        score_means.to_latex(
            os.path.join(
                args.output_dir, f"{N_SEEDS}_splits_{info_str}_score_means.tex"
            )
        )
        score_stds.to_latex(
            os.path.join(args.output_dir, f"{N_SEEDS}_splits_{info_str}_score_stds.tex")
        )
        size_means.to_latex(
            os.path.join(args.output_dir, f"{N_SEEDS}_splits_{info_str}_size_means.tex")
        )
        size_stds.to_latex(
            os.path.join(args.output_dir, f"{N_SEEDS}_splits_{info_str}_size_stds.tex")
        )
        print("Saved output to ", args.output_dir)


if __name__ == "__main__":
    main()
