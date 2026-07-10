from __future__ import annotations

from contextlib import contextmanager
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Iterator


BUILD_POINTCLOUD_TEMPLATE = "build_pointcloud_template.toml"
BUILD_SIGNAL_POINTS_TEMPLATE = "build_signal_points_template.toml"
VOXELIZE_SIGNAL_MASKS_TEMPLATE = "voxelize_signal_masks_template.toml"

TEMPLATE_FILES = {
    "build_pointcloud": BUILD_POINTCLOUD_TEMPLATE,
    "build_signal_points": BUILD_SIGNAL_POINTS_TEMPLATE,
    "voxelize_signal_masks": VOXELIZE_SIGNAL_MASKS_TEMPLATE,
}


def list_templates() -> list[str]:
    """Return supported template identifiers."""

    return sorted(TEMPLATE_FILES)


def resolve_template_filename(template_name: str) -> str:
    """Return the packaged filename for a template identifier."""

    try:
        return TEMPLATE_FILES[template_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown template '{template_name}'. Expected one of {list_templates()}."
        ) from exc


def get_template_resource(template_name: str):
    """Return the importlib resource handle for a template."""

    return importlib_resources.files(__name__).joinpath(
        resolve_template_filename(template_name)
    )


@contextmanager
def as_template_path(template_name: str) -> Iterator[Path]:
    """Yield a real filesystem path for a packaged template resource."""

    with importlib_resources.as_file(get_template_resource(template_name)) as path:
        yield path


def read_template_text(template_name: str) -> str:
    """Return the text contents of a packaged template."""

    return get_template_resource(template_name).read_text(encoding="utf-8")


def scaffold_template(
    template_name: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a packaged starter template to a user-chosen local path."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output config already exists: {output_path}. Use overwrite=True to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(read_template_text(template_name), encoding="utf-8")
    return output_path


__all__ = [
    "BUILD_POINTCLOUD_TEMPLATE",
    "BUILD_SIGNAL_POINTS_TEMPLATE",
    "TEMPLATE_FILES",
    "VOXELIZE_SIGNAL_MASKS_TEMPLATE",
    "as_template_path",
    "get_template_resource",
    "list_templates",
    "read_template_text",
    "resolve_template_filename",
    "scaffold_template",
]
