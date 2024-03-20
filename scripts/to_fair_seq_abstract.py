"""The point of this script is to create an "abstract" directory in the raw format
for fairseq. After being binarized, new datasets can be created from this data using
symbolic links.

output of write_seqs looks like:
```
ChordTonesDataSettings_settings.json  inputs_vocab.list.json                targets_0_vocab.list.pickle
OctupleEncodingSettings_settings.json inputs_vocab.list.pickle              targets_1_vocab.list.json
data                                  targets_0_vocab.list.json             targets_1_vocab.list.pickle

./data:
test  train valid

./data/test:
1.csv

./data/train:
1.csv

./data/valid:
1.csv
```

Then, if the target names are "chord_tone" and "key_pc", input to fairseq binarizer 
script should look like:

```
dict.input.txt      midi_train.txt      chord_tone_test.txt  chord_tone_valid.txt key_pc_train.txt
midi_test.txt       midi_valid.txt      chord_tone_train.txt key_pc_test.txt      key_pc_valid.txt
```

"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    seq_settings_path = input_dir / "SequenceDataSettings_settings.json"
    if not seq_settings_path.exists():
        # For backwards compatibility
        old_seq_settings_path = input_dir / "ChordTonesDataSettings_settings.json"
        if not old_seq_settings_path.exists():
            raise FileNotFoundError(seq_settings_path.name)
        seq_settings_path = old_seq_settings_path

    with open(seq_settings_path) as inf:
        seq_settings = json.load(inf)

    concat_features = ["_".join(f) for f in seq_settings["concatenated_features"]]
    targets = (
        seq_settings["features"]
        + seq_settings["sequence_level_features"]
        + concat_features
    )

    assert isinstance(targets, list)

    if os.path.exists(output_dir):
        raise ValueError(f"{output_dir=} exists")
    os.makedirs(output_dir)

    with open(output_dir / "dict.input.txt", "w") as outf:
        outf.write(INPUTS_DICT)

    for split in ("train", "test", "valid"):
        if not (input_dir / "data" / split).exists():
            print(f"Warning: {split} directory does not exist")
            continue
        csv_paths = [
            input_dir / "data" / split / f
            for f in os.listdir(input_dir / "data" / split)
            if f.endswith(".csv")
        ]
        if not csv_paths:
            print(f"Warning: no csv files found in {split}")
            continue
        for csv_path in csv_paths:
            csv_contents = pd.read_csv(csv_path)
            with open(output_dir / f"midi_{split}.txt", "a") as appendf:
                for row in csv_contents["events"]:
                    appendf.write(row)
                    appendf.write("\n")
            for target in targets:
                with open(output_dir / f"{target}_{split}.txt", "a") as appendf:
                    for row in csv_contents[target]:
                        appendf.write(str(row))
                        appendf.write("\n")
            metadata_cols = [
                name
                for name in [
                    "score_id",
                    "score_path",
                    "csv_path",
                    "transpose",
                    "scaled_by",
                    "start_offset",
                    "df_indices",
                ]
                if name in csv_contents.columns
            ]
            metadata_csv = csv_contents[metadata_cols]
            metadata_path = output_dir / f"metadata_{split}.txt"
            if metadata_path.exists():
                metadata_csv.to_csv(metadata_path, mode="a", header=False)
            else:
                metadata_csv.to_csv(metadata_path, mode="w", header=True)


# The only code past this point should be the call to main()

# INPUTS_DICT is copied directly from musicbert
INPUTS_DICT = """<0-0> 0
<0-1> 0
<0-2> 0
<0-3> 0
<0-4> 0
<0-5> 0
<0-6> 0
<0-7> 0
<0-8> 0
<0-9> 0
<0-10> 0
<0-11> 0
<0-12> 0
<0-13> 0
<0-14> 0
<0-15> 0
<0-16> 0
<0-17> 0
<0-18> 0
<0-19> 0
<0-20> 0
<0-21> 0
<0-22> 0
<0-23> 0
<0-24> 0
<0-25> 0
<0-26> 0
<0-27> 0
<0-28> 0
<0-29> 0
<0-30> 0
<0-31> 0
<0-32> 0
<0-33> 0
<0-34> 0
<0-35> 0
<0-36> 0
<0-37> 0
<0-38> 0
<0-39> 0
<0-40> 0
<0-41> 0
<0-42> 0
<0-43> 0
<0-44> 0
<0-45> 0
<0-46> 0
<0-47> 0
<0-48> 0
<0-49> 0
<0-50> 0
<0-51> 0
<0-52> 0
<0-53> 0
<0-54> 0
<0-55> 0
<0-56> 0
<0-57> 0
<0-58> 0
<0-59> 0
<0-60> 0
<0-61> 0
<0-62> 0
<0-63> 0
<0-64> 0
<0-65> 0
<0-66> 0
<0-67> 0
<0-68> 0
<0-69> 0
<0-70> 0
<0-71> 0
<0-72> 0
<0-73> 0
<0-74> 0
<0-75> 0
<0-76> 0
<0-77> 0
<0-78> 0
<0-79> 0
<0-80> 0
<0-81> 0
<0-82> 0
<0-83> 0
<0-84> 0
<0-85> 0
<0-86> 0
<0-87> 0
<0-88> 0
<0-89> 0
<0-90> 0
<0-91> 0
<0-92> 0
<0-93> 0
<0-94> 0
<0-95> 0
<0-96> 0
<0-97> 0
<0-98> 0
<0-99> 0
<0-100> 0
<0-101> 0
<0-102> 0
<0-103> 0
<0-104> 0
<0-105> 0
<0-106> 0
<0-107> 0
<0-108> 0
<0-109> 0
<0-110> 0
<0-111> 0
<0-112> 0
<0-113> 0
<0-114> 0
<0-115> 0
<0-116> 0
<0-117> 0
<0-118> 0
<0-119> 0
<0-120> 0
<0-121> 0
<0-122> 0
<0-123> 0
<0-124> 0
<0-125> 0
<0-126> 0
<0-127> 0
<0-128> 0
<0-129> 0
<0-130> 0
<0-131> 0
<0-132> 0
<0-133> 0
<0-134> 0
<0-135> 0
<0-136> 0
<0-137> 0
<0-138> 0
<0-139> 0
<0-140> 0
<0-141> 0
<0-142> 0
<0-143> 0
<0-144> 0
<0-145> 0
<0-146> 0
<0-147> 0
<0-148> 0
<0-149> 0
<0-150> 0
<0-151> 0
<0-152> 0
<0-153> 0
<0-154> 0
<0-155> 0
<0-156> 0
<0-157> 0
<0-158> 0
<0-159> 0
<0-160> 0
<0-161> 0
<0-162> 0
<0-163> 0
<0-164> 0
<0-165> 0
<0-166> 0
<0-167> 0
<0-168> 0
<0-169> 0
<0-170> 0
<0-171> 0
<0-172> 0
<0-173> 0
<0-174> 0
<0-175> 0
<0-176> 0
<0-177> 0
<0-178> 0
<0-179> 0
<0-180> 0
<0-181> 0
<0-182> 0
<0-183> 0
<0-184> 0
<0-185> 0
<0-186> 0
<0-187> 0
<0-188> 0
<0-189> 0
<0-190> 0
<0-191> 0
<0-192> 0
<0-193> 0
<0-194> 0
<0-195> 0
<0-196> 0
<0-197> 0
<0-198> 0
<0-199> 0
<0-200> 0
<0-201> 0
<0-202> 0
<0-203> 0
<0-204> 0
<0-205> 0
<0-206> 0
<0-207> 0
<0-208> 0
<0-209> 0
<0-210> 0
<0-211> 0
<0-212> 0
<0-213> 0
<0-214> 0
<0-215> 0
<0-216> 0
<0-217> 0
<0-218> 0
<0-219> 0
<0-220> 0
<0-221> 0
<0-222> 0
<0-223> 0
<0-224> 0
<0-225> 0
<0-226> 0
<0-227> 0
<0-228> 0
<0-229> 0
<0-230> 0
<0-231> 0
<0-232> 0
<0-233> 0
<0-234> 0
<0-235> 0
<0-236> 0
<0-237> 0
<0-238> 0
<0-239> 0
<0-240> 0
<0-241> 0
<0-242> 0
<0-243> 0
<0-244> 0
<0-245> 0
<0-246> 0
<0-247> 0
<0-248> 0
<0-249> 0
<0-250> 0
<0-251> 0
<0-252> 0
<0-253> 0
<0-254> 0
<0-255> 0
<1-0> 0
<1-1> 0
<1-2> 0
<1-3> 0
<1-4> 0
<1-5> 0
<1-6> 0
<1-7> 0
<1-8> 0
<1-9> 0
<1-10> 0
<1-11> 0
<1-12> 0
<1-13> 0
<1-14> 0
<1-15> 0
<1-16> 0
<1-17> 0
<1-18> 0
<1-19> 0
<1-20> 0
<1-21> 0
<1-22> 0
<1-23> 0
<1-24> 0
<1-25> 0
<1-26> 0
<1-27> 0
<1-28> 0
<1-29> 0
<1-30> 0
<1-31> 0
<1-32> 0
<1-33> 0
<1-34> 0
<1-35> 0
<1-36> 0
<1-37> 0
<1-38> 0
<1-39> 0
<1-40> 0
<1-41> 0
<1-42> 0
<1-43> 0
<1-44> 0
<1-45> 0
<1-46> 0
<1-47> 0
<1-48> 0
<1-49> 0
<1-50> 0
<1-51> 0
<1-52> 0
<1-53> 0
<1-54> 0
<1-55> 0
<1-56> 0
<1-57> 0
<1-58> 0
<1-59> 0
<1-60> 0
<1-61> 0
<1-62> 0
<1-63> 0
<1-64> 0
<1-65> 0
<1-66> 0
<1-67> 0
<1-68> 0
<1-69> 0
<1-70> 0
<1-71> 0
<1-72> 0
<1-73> 0
<1-74> 0
<1-75> 0
<1-76> 0
<1-77> 0
<1-78> 0
<1-79> 0
<1-80> 0
<1-81> 0
<1-82> 0
<1-83> 0
<1-84> 0
<1-85> 0
<1-86> 0
<1-87> 0
<1-88> 0
<1-89> 0
<1-90> 0
<1-91> 0
<1-92> 0
<1-93> 0
<1-94> 0
<1-95> 0
<1-96> 0
<1-97> 0
<1-98> 0
<1-99> 0
<1-100> 0
<1-101> 0
<1-102> 0
<1-103> 0
<1-104> 0
<1-105> 0
<1-106> 0
<1-107> 0
<1-108> 0
<1-109> 0
<1-110> 0
<1-111> 0
<1-112> 0
<1-113> 0
<1-114> 0
<1-115> 0
<1-116> 0
<1-117> 0
<1-118> 0
<1-119> 0
<1-120> 0
<1-121> 0
<1-122> 0
<1-123> 0
<1-124> 0
<1-125> 0
<1-126> 0
<1-127> 0
<2-0> 0
<2-1> 0
<2-2> 0
<2-3> 0
<2-4> 0
<2-5> 0
<2-6> 0
<2-7> 0
<2-8> 0
<2-9> 0
<2-10> 0
<2-11> 0
<2-12> 0
<2-13> 0
<2-14> 0
<2-15> 0
<2-16> 0
<2-17> 0
<2-18> 0
<2-19> 0
<2-20> 0
<2-21> 0
<2-22> 0
<2-23> 0
<2-24> 0
<2-25> 0
<2-26> 0
<2-27> 0
<2-28> 0
<2-29> 0
<2-30> 0
<2-31> 0
<2-32> 0
<2-33> 0
<2-34> 0
<2-35> 0
<2-36> 0
<2-37> 0
<2-38> 0
<2-39> 0
<2-40> 0
<2-41> 0
<2-42> 0
<2-43> 0
<2-44> 0
<2-45> 0
<2-46> 0
<2-47> 0
<2-48> 0
<2-49> 0
<2-50> 0
<2-51> 0
<2-52> 0
<2-53> 0
<2-54> 0
<2-55> 0
<2-56> 0
<2-57> 0
<2-58> 0
<2-59> 0
<2-60> 0
<2-61> 0
<2-62> 0
<2-63> 0
<2-64> 0
<2-65> 0
<2-66> 0
<2-67> 0
<2-68> 0
<2-69> 0
<2-70> 0
<2-71> 0
<2-72> 0
<2-73> 0
<2-74> 0
<2-75> 0
<2-76> 0
<2-77> 0
<2-78> 0
<2-79> 0
<2-80> 0
<2-81> 0
<2-82> 0
<2-83> 0
<2-84> 0
<2-85> 0
<2-86> 0
<2-87> 0
<2-88> 0
<2-89> 0
<2-90> 0
<2-91> 0
<2-92> 0
<2-93> 0
<2-94> 0
<2-95> 0
<2-96> 0
<2-97> 0
<2-98> 0
<2-99> 0
<2-100> 0
<2-101> 0
<2-102> 0
<2-103> 0
<2-104> 0
<2-105> 0
<2-106> 0
<2-107> 0
<2-108> 0
<2-109> 0
<2-110> 0
<2-111> 0
<2-112> 0
<2-113> 0
<2-114> 0
<2-115> 0
<2-116> 0
<2-117> 0
<2-118> 0
<2-119> 0
<2-120> 0
<2-121> 0
<2-122> 0
<2-123> 0
<2-124> 0
<2-125> 0
<2-126> 0
<2-127> 0
<2-128> 0
<3-0> 0
<3-1> 0
<3-2> 0
<3-3> 0
<3-4> 0
<3-5> 0
<3-6> 0
<3-7> 0
<3-8> 0
<3-9> 0
<3-10> 0
<3-11> 0
<3-12> 0
<3-13> 0
<3-14> 0
<3-15> 0
<3-16> 0
<3-17> 0
<3-18> 0
<3-19> 0
<3-20> 0
<3-21> 0
<3-22> 0
<3-23> 0
<3-24> 0
<3-25> 0
<3-26> 0
<3-27> 0
<3-28> 0
<3-29> 0
<3-30> 0
<3-31> 0
<3-32> 0
<3-33> 0
<3-34> 0
<3-35> 0
<3-36> 0
<3-37> 0
<3-38> 0
<3-39> 0
<3-40> 0
<3-41> 0
<3-42> 0
<3-43> 0
<3-44> 0
<3-45> 0
<3-46> 0
<3-47> 0
<3-48> 0
<3-49> 0
<3-50> 0
<3-51> 0
<3-52> 0
<3-53> 0
<3-54> 0
<3-55> 0
<3-56> 0
<3-57> 0
<3-58> 0
<3-59> 0
<3-60> 0
<3-61> 0
<3-62> 0
<3-63> 0
<3-64> 0
<3-65> 0
<3-66> 0
<3-67> 0
<3-68> 0
<3-69> 0
<3-70> 0
<3-71> 0
<3-72> 0
<3-73> 0
<3-74> 0
<3-75> 0
<3-76> 0
<3-77> 0
<3-78> 0
<3-79> 0
<3-80> 0
<3-81> 0
<3-82> 0
<3-83> 0
<3-84> 0
<3-85> 0
<3-86> 0
<3-87> 0
<3-88> 0
<3-89> 0
<3-90> 0
<3-91> 0
<3-92> 0
<3-93> 0
<3-94> 0
<3-95> 0
<3-96> 0
<3-97> 0
<3-98> 0
<3-99> 0
<3-100> 0
<3-101> 0
<3-102> 0
<3-103> 0
<3-104> 0
<3-105> 0
<3-106> 0
<3-107> 0
<3-108> 0
<3-109> 0
<3-110> 0
<3-111> 0
<3-112> 0
<3-113> 0
<3-114> 0
<3-115> 0
<3-116> 0
<3-117> 0
<3-118> 0
<3-119> 0
<3-120> 0
<3-121> 0
<3-122> 0
<3-123> 0
<3-124> 0
<3-125> 0
<3-126> 0
<3-127> 0
<3-128> 0
<3-129> 0
<3-130> 0
<3-131> 0
<3-132> 0
<3-133> 0
<3-134> 0
<3-135> 0
<3-136> 0
<3-137> 0
<3-138> 0
<3-139> 0
<3-140> 0
<3-141> 0
<3-142> 0
<3-143> 0
<3-144> 0
<3-145> 0
<3-146> 0
<3-147> 0
<3-148> 0
<3-149> 0
<3-150> 0
<3-151> 0
<3-152> 0
<3-153> 0
<3-154> 0
<3-155> 0
<3-156> 0
<3-157> 0
<3-158> 0
<3-159> 0
<3-160> 0
<3-161> 0
<3-162> 0
<3-163> 0
<3-164> 0
<3-165> 0
<3-166> 0
<3-167> 0
<3-168> 0
<3-169> 0
<3-170> 0
<3-171> 0
<3-172> 0
<3-173> 0
<3-174> 0
<3-175> 0
<3-176> 0
<3-177> 0
<3-178> 0
<3-179> 0
<3-180> 0
<3-181> 0
<3-182> 0
<3-183> 0
<3-184> 0
<3-185> 0
<3-186> 0
<3-187> 0
<3-188> 0
<3-189> 0
<3-190> 0
<3-191> 0
<3-192> 0
<3-193> 0
<3-194> 0
<3-195> 0
<3-196> 0
<3-197> 0
<3-198> 0
<3-199> 0
<3-200> 0
<3-201> 0
<3-202> 0
<3-203> 0
<3-204> 0
<3-205> 0
<3-206> 0
<3-207> 0
<3-208> 0
<3-209> 0
<3-210> 0
<3-211> 0
<3-212> 0
<3-213> 0
<3-214> 0
<3-215> 0
<3-216> 0
<3-217> 0
<3-218> 0
<3-219> 0
<3-220> 0
<3-221> 0
<3-222> 0
<3-223> 0
<3-224> 0
<3-225> 0
<3-226> 0
<3-227> 0
<3-228> 0
<3-229> 0
<3-230> 0
<3-231> 0
<3-232> 0
<3-233> 0
<3-234> 0
<3-235> 0
<3-236> 0
<3-237> 0
<3-238> 0
<3-239> 0
<3-240> 0
<3-241> 0
<3-242> 0
<3-243> 0
<3-244> 0
<3-245> 0
<3-246> 0
<3-247> 0
<3-248> 0
<3-249> 0
<3-250> 0
<3-251> 0
<3-252> 0
<3-253> 0
<3-254> 0
<3-255> 0
<4-0> 0
<4-1> 0
<4-2> 0
<4-3> 0
<4-4> 0
<4-5> 0
<4-6> 0
<4-7> 0
<4-8> 0
<4-9> 0
<4-10> 0
<4-11> 0
<4-12> 0
<4-13> 0
<4-14> 0
<4-15> 0
<4-16> 0
<4-17> 0
<4-18> 0
<4-19> 0
<4-20> 0
<4-21> 0
<4-22> 0
<4-23> 0
<4-24> 0
<4-25> 0
<4-26> 0
<4-27> 0
<4-28> 0
<4-29> 0
<4-30> 0
<4-31> 0
<4-32> 0
<4-33> 0
<4-34> 0
<4-35> 0
<4-36> 0
<4-37> 0
<4-38> 0
<4-39> 0
<4-40> 0
<4-41> 0
<4-42> 0
<4-43> 0
<4-44> 0
<4-45> 0
<4-46> 0
<4-47> 0
<4-48> 0
<4-49> 0
<4-50> 0
<4-51> 0
<4-52> 0
<4-53> 0
<4-54> 0
<4-55> 0
<4-56> 0
<4-57> 0
<4-58> 0
<4-59> 0
<4-60> 0
<4-61> 0
<4-62> 0
<4-63> 0
<4-64> 0
<4-65> 0
<4-66> 0
<4-67> 0
<4-68> 0
<4-69> 0
<4-70> 0
<4-71> 0
<4-72> 0
<4-73> 0
<4-74> 0
<4-75> 0
<4-76> 0
<4-77> 0
<4-78> 0
<4-79> 0
<4-80> 0
<4-81> 0
<4-82> 0
<4-83> 0
<4-84> 0
<4-85> 0
<4-86> 0
<4-87> 0
<4-88> 0
<4-89> 0
<4-90> 0
<4-91> 0
<4-92> 0
<4-93> 0
<4-94> 0
<4-95> 0
<4-96> 0
<4-97> 0
<4-98> 0
<4-99> 0
<4-100> 0
<4-101> 0
<4-102> 0
<4-103> 0
<4-104> 0
<4-105> 0
<4-106> 0
<4-107> 0
<4-108> 0
<4-109> 0
<4-110> 0
<4-111> 0
<4-112> 0
<4-113> 0
<4-114> 0
<4-115> 0
<4-116> 0
<4-117> 0
<4-118> 0
<4-119> 0
<4-120> 0
<4-121> 0
<4-122> 0
<4-123> 0
<4-124> 0
<4-125> 0
<4-126> 0
<4-127> 0
<5-0> 0
<5-1> 0
<5-2> 0
<5-3> 0
<5-4> 0
<5-5> 0
<5-6> 0
<5-7> 0
<5-8> 0
<5-9> 0
<5-10> 0
<5-11> 0
<5-12> 0
<5-13> 0
<5-14> 0
<5-15> 0
<5-16> 0
<5-17> 0
<5-18> 0
<5-19> 0
<5-20> 0
<5-21> 0
<5-22> 0
<5-23> 0
<5-24> 0
<5-25> 0
<5-26> 0
<5-27> 0
<5-28> 0
<5-29> 0
<5-30> 0
<5-31> 0
<6-0> 0
<6-1> 0
<6-2> 0
<6-3> 0
<6-4> 0
<6-5> 0
<6-6> 0
<6-7> 0
<6-8> 0
<6-9> 0
<6-10> 0
<6-11> 0
<6-12> 0
<6-13> 0
<6-14> 0
<6-15> 0
<6-16> 0
<6-17> 0
<6-18> 0
<6-19> 0
<6-20> 0
<6-21> 0
<6-22> 0
<6-23> 0
<6-24> 0
<6-25> 0
<6-26> 0
<6-27> 0
<6-28> 0
<6-29> 0
<6-30> 0
<6-31> 0
<6-32> 0
<6-33> 0
<6-34> 0
<6-35> 0
<6-36> 0
<6-37> 0
<6-38> 0
<6-39> 0
<6-40> 0
<6-41> 0
<6-42> 0
<6-43> 0
<6-44> 0
<6-45> 0
<6-46> 0
<6-47> 0
<6-48> 0
<6-49> 0
<6-50> 0
<6-51> 0
<6-52> 0
<6-53> 0
<6-54> 0
<6-55> 0
<6-56> 0
<6-57> 0
<6-58> 0
<6-59> 0
<6-60> 0
<6-61> 0
<6-62> 0
<6-63> 0
<6-64> 0
<6-65> 0
<6-66> 0
<6-67> 0
<6-68> 0
<6-69> 0
<6-70> 0
<6-71> 0
<6-72> 0
<6-73> 0
<6-74> 0
<6-75> 0
<6-76> 0
<6-77> 0
<6-78> 0
<6-79> 0
<6-80> 0
<6-81> 0
<6-82> 0
<6-83> 0
<6-84> 0
<6-85> 0
<6-86> 0
<6-87> 0
<6-88> 0
<6-89> 0
<6-90> 0
<6-91> 0
<6-92> 0
<6-93> 0
<6-94> 0
<6-95> 0
<6-96> 0
<6-97> 0
<6-98> 0
<6-99> 0
<6-100> 0
<6-101> 0
<6-102> 0
<6-103> 0
<6-104> 0
<6-105> 0
<6-106> 0
<6-107> 0
<6-108> 0
<6-109> 0
<6-110> 0
<6-111> 0
<6-112> 0
<6-113> 0
<6-114> 0
<6-115> 0
<6-116> 0
<6-117> 0
<6-118> 0
<6-119> 0
<6-120> 0
<6-121> 0
<6-122> 0
<6-123> 0
<6-124> 0
<6-125> 0
<6-126> 0
<6-127> 0
<6-128> 0
<6-129> 0
<6-130> 0
<6-131> 0
<6-132> 0
<6-133> 0
<6-134> 0
<6-135> 0
<6-136> 0
<6-137> 0
<6-138> 0
<6-139> 0
<6-140> 0
<6-141> 0
<6-142> 0
<6-143> 0
<6-144> 0
<6-145> 0
<6-146> 0
<6-147> 0
<6-148> 0
<6-149> 0
<6-150> 0
<6-151> 0
<6-152> 0
<6-153> 0
<6-154> 0
<6-155> 0
<6-156> 0
<6-157> 0
<6-158> 0
<6-159> 0
<6-160> 0
<6-161> 0
<6-162> 0
<6-163> 0
<6-164> 0
<6-165> 0
<6-166> 0
<6-167> 0
<6-168> 0
<6-169> 0
<6-170> 0
<6-171> 0
<6-172> 0
<6-173> 0
<6-174> 0
<6-175> 0
<6-176> 0
<6-177> 0
<6-178> 0
<6-179> 0
<6-180> 0
<6-181> 0
<6-182> 0
<6-183> 0
<6-184> 0
<6-185> 0
<6-186> 0
<6-187> 0
<6-188> 0
<6-189> 0
<6-190> 0
<6-191> 0
<6-192> 0
<6-193> 0
<6-194> 0
<6-195> 0
<6-196> 0
<6-197> 0
<6-198> 0
<6-199> 0
<6-200> 0
<6-201> 0
<6-202> 0
<6-203> 0
<6-204> 0
<6-205> 0
<6-206> 0
<6-207> 0
<6-208> 0
<6-209> 0
<6-210> 0
<6-211> 0
<6-212> 0
<6-213> 0
<6-214> 0
<6-215> 0
<6-216> 0
<6-217> 0
<6-218> 0
<6-219> 0
<6-220> 0
<6-221> 0
<6-222> 0
<6-223> 0
<6-224> 0
<6-225> 0
<6-226> 0
<6-227> 0
<6-228> 0
<6-229> 0
<6-230> 0
<6-231> 0
<6-232> 0
<6-233> 0
<6-234> 0
<6-235> 0
<6-236> 0
<6-237> 0
<6-238> 0
<6-239> 0
<6-240> 0
<6-241> 0
<6-242> 0
<6-243> 0
<6-244> 0
<6-245> 0
<6-246> 0
<6-247> 0
<6-248> 0
<6-249> 0
<6-250> 0
<6-251> 0
<6-252> 0
<6-253> 0
<7-0> 0
<7-1> 0
<7-2> 0
<7-3> 0
<7-4> 0
<7-5> 0
<7-6> 0
<7-7> 0
<7-8> 0
<7-9> 0
<7-10> 0
<7-11> 0
<7-12> 0
<7-13> 0
<7-14> 0
<7-15> 0
<7-16> 0
<7-17> 0
<7-18> 0
<7-19> 0
<7-20> 0
<7-21> 0
<7-22> 0
<7-23> 0
<7-24> 0
<7-25> 0
<7-26> 0
<7-27> 0
<7-28> 0
<7-29> 0
<7-30> 0
<7-31> 0
<7-32> 0
<7-33> 0
<7-34> 0
<7-35> 0
<7-36> 0
<7-37> 0
<7-38> 0
<7-39> 0
<7-40> 0
<7-41> 0
<7-42> 0
<7-43> 0
<7-44> 0
<7-45> 0
<7-46> 0
<7-47> 0
<7-48> 0
"""

if __name__ == "__main__":
    main()
