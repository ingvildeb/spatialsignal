"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def load_toml_config(config_path: Path) -> dict[str, Any]:
    """Load a TOML config file into a dictionary."""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {config_path}")

    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def require_config_value(config: dict[str, Any], section: str, key: str) -> Any:
    """Return a required config value or raise a clear error."""

    if section not in config:
        raise KeyError(f"Missing config section: [{section}]")

    section_data = config[section]
    if key not in section_data:
        raise KeyError(f"Missing config value: [{section}] {key}")

    return section_data[key]
