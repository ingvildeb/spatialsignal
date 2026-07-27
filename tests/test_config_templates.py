from pathlib import Path

from spatialsignal.config_templates import (
    list_templates,
    read_template_text,
    scaffold_template,
)


def test_list_templates_returns_expected_ids() -> None:
    assert list_templates() == [
        "build_pointcloud",
        "build_signal_points",
        "colocalization",
        "voxelize_signal_masks",
    ]


def test_read_template_text_contains_zero_based_indexing() -> None:
    text = read_template_text("build_pointcloud")
    assert 'indexing = "zero_based"' in text


def test_scaffold_template_writes_requested_file(tmp_path: Path) -> None:
    output_path = tmp_path / "build_pointcloud_local.toml"

    written_path = scaffold_template("build_pointcloud", output_path)

    assert written_path == output_path
    assert output_path.exists()
    assert 'name = "Example_Subject"' in output_path.read_text(encoding="utf-8")
