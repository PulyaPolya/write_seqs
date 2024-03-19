import json
import os
import typing as t
from dataclasses import asdict, dataclass, field

from write_seqs.utils.get_hash import get_hash

from reprs.midi_like import MidiLikeSettings
from reprs.oct import OctupleEncodingSettings
from reprs.shared import ReprSettingsBase


def get_dataset_base_dir():
    # I encapsulate this in a function so we can change the environment variable
    #   in mock tests
    return os.getenv("WRITE_SEQS_BASE_DIR", "")


@dataclass()
class SequenceDataSettings:
    # NB: for MidiLikeEncoding, if "keep_onsets_together", hop is measured in "unique
    #   onsets"; there could be 6 notes sounding at time 1, but hop only considers them
    #   as one onset.
    features: t.List[str] = field(default_factory=list)
    concatenated_features: t.List[t.Sequence[str]] = field(default_factory=list)

    # we look for sequence_level_features in the df attrs
    sequence_level_features: t.List[str] = field(default_factory=list)
    hop: int = 8
    window_len: t.Optional[int] = 128
    min_window_len: t.Optional[int] = None
    aug_synthetic_data: bool = False
    aug_by_key: bool = False
    aug_by_key_n_keys: int = 12
    aug_within_range: bool = False
    aug_within_range_n_keys: int = 12
    aug_rhythms: bool = False
    aug_rhythms_n_augs: int = 1
    aug_rhythms_n_possibilities: int = 2
    drop_spelling: bool = False

    repr_type: t.Literal["oct", "midilike"] = "midilike"

    dataset_name: t.Optional[str] = None

    corpora_to_exclude: t.List[str] = field(default_factory=list)
    # corpora_to_include is ignored if it is empty, otherwise it takes
    #   precedence over corpora_to_sclude
    corpora_to_include: t.List[str] = field(default_factory=list)
    # synthetic corpora must be explicitly included, otherwise they are excluded
    synthetic_corpora_to_include: t.List[str] = field(default_factory=list)
    training_only_corpora: t.List[str] = field(default_factory=list)
    corpora_sample_proportions: t.Dict[str, float] = field(
        default_factory=lambda: {"RenDissData": 0.1}
    )
    split_seed: t.Optional[int] = 42
    split_by_corpora: bool = True
    proportions_exclude_training_only_items: bool = True
    use_tempi: bool = True
    train_paths_to_include: t.Optional[str] = None
    valid_paths_to_include: t.Optional[str] = None
    test_paths_to_include: t.Optional[str] = None

    use_existing_splits: bool = True

    repr_settings_oct: OctupleEncodingSettings = field(
        default_factory=OctupleEncodingSettings
    )
    repr_settings_midilike: MidiLikeSettings = field(default_factory=MidiLikeSettings)

    def __post_init__(self):
        if isinstance(self.features, str):
            self.features = [self.features]
        if isinstance(self.sequence_level_features, str):
            self.sequence_level_features = [self.sequence_level_features]
        if isinstance(self.training_only_corpora, str):
            self.training_only_corpora = [self.training_only_corpora]

        for f in self.concatenated_features:
            assert not isinstance(f, str)

    @property
    def repr_settings(self) -> ReprSettingsBase:
        if self.repr_type == "oct":
            return self.repr_settings_oct
        elif self.repr_type == "midilike":
            return self.repr_settings_midilike
        else:
            raise ValueError

    @repr_settings.setter
    def repr_settings(self, value):
        if self.repr_type == "oct":
            assert isinstance(value, OctupleEncodingSettings)
            self.repr_settings_oct = value
        elif self.repr_type == "midilike":
            assert isinstance(value, MidiLikeSettings)
            self.repr_settings_midilike = value
        else:
            raise ValueError


def save_dclass(dclass, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    outpath = os.path.join(output_folder, f"{dclass.__class__.__name__}_settings.json")
    with open(outpath, "w") as outf:
        json.dump(asdict(dclass), outf, indent=4)


# def path_from_dataclass(dclass, base_dir=None, **kwargs):
#     if "test" in kwargs and not kwargs["test"]:
#         del kwargs["test"]
#     null_args = []
#     for kwarg, val in kwargs.items():
#         if isinstance(val, bool):
#             kwargs[kwarg] = int(val)
#         elif isinstance(val, (list, tuple)):
#             kwargs[kwarg] = "+".join(str(item) for item in val)
#         elif isinstance(val, type(None)):
#             null_args.append(kwarg)
#     for kwarg in null_args:
#         del kwargs[kwarg]
#     kwarg_str = "_".join(f'{k.replace("_", "-")}={v}' for k, v in kwargs.items())
#     if base_dir is None:
#         path_components = [get_dataset_dir(), __package__]
#     else:
#         path_components = [base_dir]
#     if hasattr(dclass, "dataset_name") and dclass.dataset_name:
#         path_components.append(dclass.dataset_name)
#     path_components.extend([str(get_hash(dclass)), kwarg_str])
#     return os.path.join(*path_components)
