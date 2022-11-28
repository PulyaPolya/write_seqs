#!/usr/bin/env bash

echo "Warning: if you include --data-args argument it will override the value"
echo    "in this script."

set -x
python3 -m write_chord_tones_seqs \
    --data-args corpora_to_include=ABCData,BpsFhData,MozartPSData,TavernData,HaydnQsData
set +x
