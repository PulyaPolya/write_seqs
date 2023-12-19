import json
import os
import typing as t
from dataclasses import asdict, dataclass, field

from write_seqs.utils.get_hash import get_hash


def get_dataset_base_dir():
    # I encapsulate this in a function so we can change the environment variable
    #   in mock tests
    return os.getenv("WRITE_SEQS_BASE_DIR", "")


@dataclass()
class SequenceDataSettings:
    # NB: for MidiLikeEncoding, if "keep_onsets_together", hop is measured in "unique
    #   onsets"; there could be 6 notes sounding at time 1, but hop only considers them
    #   as one onset.
    features: t.Sequence[str]
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

    repr_type: t.Literal["oct", "midilike"] = "midilike"

    dataset_name: t.Optional[str] = None

    corpora_to_exclude: t.Sequence[str] = ()
    # corpora_to_include is ignored if it is empty, otherwise it takes
    #   precedence over corpora_to_sclude
    corpora_to_include: t.Sequence[str] = ()
    # synthetic corpora must be explicitly included, otherwise they are excluded
    synthetic_corpora_to_include: t.Sequence[str] = ()
    training_only_corpora: t.Union[str, t.Sequence[str]] = ()
    corpora_sample_proportions: t.Dict[str, float] = field(
        default_factory=lambda: {"RenDissData": 0.1}
    )
    split_seed: t.Optional[int] = 42
    split_by_corpora: bool = True
    proportions_exclude_training_only_items: bool = True
    use_tempi: bool = True

    def __post_init__(self):
        if isinstance(self.features, str):
            self.features = (self.features,)
        if isinstance(self.training_only_corpora, str):
            self.training_only_corpora = (self.training_only_corpora,)


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
