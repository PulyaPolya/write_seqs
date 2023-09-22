Take a corpus of "music_df" and convert to sequences of notes and features.


Expects the following format
- parent directory (e.g., `/Users/malcolm/output/chord_tones_datasets/salami_slice_no_suspensions`) contains
    - vocabularies: json files containing lists of strings formatted as `{feature_name}_vocab.json`
    - corpora: subfolders (e.g., `ABCData/`) containing:
        - an `attrs.json` file containing (TODO)
        - paired csv and json files for each score, e.g., `n01op18-1_01.csv` and `n01op18-1_01.json`
        - the csv file contains the "music_df" and features TODO 2023-09-22 describe obligatory columns 
        - the json contains "attributes" of the score TODO specify
