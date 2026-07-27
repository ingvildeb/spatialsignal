from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from spatialsignal.pointcloud import match_colocalized_detections
from spatialsignal.qc import write_colocalization_images, write_colocalization_report


def _dataset(rows: list[dict[str, int]]) -> PointCloudDataset:
    points = pd.DataFrame(rows, columns=["detection_id", "seg_num", "x", "y", "z"])
    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="native",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="zero_based",
            units="voxel",
            shape=[20, 10, 3],
            resolution_um=[1.0, 1.0, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )
    return PointCloudDataset("subject", points, metadata)


def test_write_colocalization_images_writes_every_plane_at_native_shape(
    tmp_path: Path,
) -> None:
    dataset_a = _dataset(
        [
            {"detection_id": 1, "seg_num": 1, "x": 5, "y": 4, "z": 0},
            {"detection_id": 2, "seg_num": 2, "x": 10, "y": 4, "z": 1},
        ]
    )
    dataset_b = _dataset(
        [{"detection_id": 11, "seg_num": 1, "x": 6, "y": 4, "z": 0}]
    )
    result = match_colocalized_detections(
        dataset_a,
        dataset_b,
        max_xy_distance_um=2.0,
    )
    stale_path = tmp_path / "qc_images" / "subject_colocalization_z9999.png"
    stale_path.parent.mkdir()
    stale_path.touch()

    paths = write_colocalization_images(
        dataset_a,
        dataset_b,
        result,
        tmp_path / "qc_images",
        output_stem="subject",
    )

    assert set(paths) == {0, 1, 2}
    assert not stale_path.exists()
    assert all(path.exists() for path in paths.values())
    image = Image.open(paths[0])
    assert image.size == (20, 10)
    assert image.getchannel("A").getextrema()[1] == 255
    assert Image.open(paths[1]).getchannel("A").getextrema()[1] == 55
    assert Image.open(paths[2]).getchannel("A").getextrema()[1] == 0


def test_write_colocalization_report_links_every_png(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    summary = pd.DataFrame(
        [
            {
                "z": 0,
                "n_a_detections": 1,
                "n_b_detections": 1,
                "n_candidate_pairs": 1,
                "n_accepted_matches": 1,
                "n_unmatched_a": 0,
                "n_unmatched_b": 0,
                "raw_match_rate_a": 1.0,
                "raw_match_rate_b": 1.0,
                "median_match_distance_um": 1.0,
                "maximum_match_distance_um": 1.0,
                "n_ambiguous_a": 0,
                "n_ambiguous_b": 0,
            }
        ]
    )
    image_path = tmp_path / "qc_images" / "subject_z0000.png"
    image_path.parent.mkdir()
    image_path.touch()
    workbook_path = write_colocalization_report(
        summary,
        {0: image_path},
        tmp_path / "subject_colocalization_qc.xlsx",
        channel_a_name="ch0_GFP",
        channel_b_name="ch1_tdTomato",
    )

    workbook = openpyxl.load_workbook(workbook_path)
    worksheet = workbook["planes"]
    assert worksheet.max_row == 2
    headers = [cell.value for cell in worksheet[1]]
    assert "n_ch0_GFP_detections" in headers
    assert "n_ch1_tdTomato_detections" in headers
    assert "n_a_detections" not in headers
    assert "n_b_detections" not in headers
    png_column = headers.index("png_file") + 1
    png_cell = worksheet.cell(row=2, column=png_column)
    assert png_cell.value == "qc_images/subject_z0000.png"
    assert png_cell.hyperlink.target == "qc_images/subject_z0000.png"
