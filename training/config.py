"""Configuration helpers shared by A1 command-line tools."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"A1 config does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    if config.get("stage") != "A1":
        raise ValueError("Only stage A1 is supported by this training entrypoint.")
    return config


def resolve_path(config_path: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent.parent / path).resolve()
