Take a corpus of "music_df" and convert to sequences of notes and features.


Expects the following format
- parent directory (e.g., `/Users/malcolm/datasets/chord_tones/salami_slice_no_suspensions`) contains
    - optional vocabularies: json files containing lists of strings formatted as `{feature_name}_vocab.json`
        - these can probably be omitted without consequence at the moment; a warning will be emitted.
    - corpora: subfolders (e.g., `ABCData/`) containing:
        - an `attrs.json` file containing attributes of the corpus. This file is optional, however the representation types look for this file to validate whether they can represent the given corpus. See below.
        - paired csv and json files for each score, e.g., `n01op18-1_01.csv` and `n01op18-1_01.json`
        - the csv file contains the "music_df" and features TODO 2023-09-22 describe obligatory columns 
        - the json contains "attributes" of the score TODO specify

# Input

The input directory should contain one or more subdirectories, each of which contains a "corpus".

Each subdirectory has the following contents:
1. An `attrs.json` file. For the purposes of the octuple encoding, this file can have the following contents:
```json
{
    "has_time_signatures": true
}
```
2. CSV files, one per score, with the structure described below.
3. Optional per-score JSON files. These should have the same name as the CSV file, minus the extension. <!-- TODO 2023-09-27 describe further -->

## CSV files

Each CSV file should have the following columns: "onset", "release", "type", "pitch", "other", and at least one more column that contains the feature you want to predict, which you can call whatever you like.

This file should contain at least three types of events:
1. `time_signature`. These should have an "onset", and in the "other" column they should have the time signature formatted as a Python dictionary, like `"{""numerator"": 4, ""denominator"": 4}"`. (Here, I've escaped the quotes in the default Pandas CSV way; if you use Pandas to make the CSV file it should do this automatically.) There should be a time signature at the start of the score.
2. At least one `bar` event, at the beginning of the store. If you don't add other bars explicitly they will be inferred from the time signature. This should have an onset and a release.
3. `note` events. These have onset, release, and pitch. 

Here's an example:

```csv
onset,release,type,pitch,other,dummy_feature
0.0,4.0,bar,,,na
0.0,,time_signature,,"{""numerator"": 4, ""denominator"": 4}",na
0.0,1.0,note,40.0,,foo
1.0,2.0,note,42.0,,bar
2.0,4.0,note,44.0,,foo
```


## Required attributes per representation type:

### Octuple encoding

`attrs.json` must contain `has_time_signatures: true`

### Midi-like encoding

If `include_metric_weights` is True, `attrs.json` must contain `has_weights: true`.
