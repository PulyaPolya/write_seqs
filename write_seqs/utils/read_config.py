import argparse
from omegaconf import OmegaConf
from dacite import from_dict


def read_config_oc(
    *, config_cls, config_path: str | None = None, cli_args: list[str] | None = None
):
    if config_path is None and cli_args is None:
        return config_cls()

    configs = []
    if config_path is not None:
        configs.append(OmegaConf.load(config_path))
    if cli_args is not None:
        configs.append(OmegaConf.from_cli(cli_args))
    merged_conf = OmegaConf.merge(*configs)
    resolved = OmegaConf.to_container(merged_conf, resolve=True)
    assert isinstance(resolved, dict)
    out = from_dict(data_class=config_cls, data=resolved)  # type:ignore
    return out
