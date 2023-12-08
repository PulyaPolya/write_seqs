import os
from collections import Counter, defaultdict
from itertools import chain

import pandas as pd


def conduct_census(dir_name: str) -> pd.DataFrame:
    splits = (
        "train",
        "valid",
        "test",
    )
    dict_accumulator = []
    col_names = []
    for split in splits:
        split_dir = os.path.join(dir_name, split)
        if not os.path.exists(split_dir):
            continue
        example_counter = Counter()
        unique_scores = defaultdict(set)
        df_paths = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".csv")
        ]
        for df_path in df_paths:
            sub_df = pd.read_csv(df_path)
            for p in sub_df.csv_path:
                csv_dir, score_name = os.path.split(p)
                corpus_name = os.path.basename(csv_dir)
                example_counter[corpus_name] += 1
                unique_scores[corpus_name].add(score_name)

        dict_accumulator.append(dict(example_counter))
        dict_accumulator.append(
            {corpus_name: len(scores) for corpus_name, scores in unique_scores.items()}
        )
        col_names.append(split)

    out_df = pd.DataFrame(dict_accumulator).transpose()
    out_df = out_df.fillna(value=0)
    example_col_names = [f"Num {split} examples" for split in col_names]
    score_col_names = [f"Num {split} unique scores" for split in col_names]

    out_df.columns = list(chain(*zip(example_col_names, score_col_names)))
    out_df["total_examples"] = out_df[example_col_names].sum(axis=1)
    out_df["total_unique_scores"] = out_df[score_col_names].sum(axis=1)
    for col in out_df.columns:
        out_df[col] = out_df[col].astype(int)
    out_df.loc["total"] = out_df.sum()
    return out_df
