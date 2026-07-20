"""Tests for canonical dataset output formats."""

from pathlib import Path

import numpy as np
import pandas as pd

from spatialsignal.integration import LabelVolume
from spatialsignal.io import (
    save_deduplication_outputs,
    save_instance_region_quantification_outputs,
)
from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from spatialsignal.pointcloud import DeduplicationResult


def _metadata() -> DatasetMetadata:
    return DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="zero_based",
            units="voxel",
            shape=[10, 10, 2],
            resolution_um=[1.0, 1.0, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )


def test_large_computational_tables_are_saved_as_parquet(tmp_path: Path) -> None:
    points = pd.DataFrame(
        [{"detection_id": 1, "seg_num": 1, "x": 2, "y": 3, "z": 0}]
    )
    dataset = PointCloudDataset("subject_1", points, _metadata())
    result = DeduplicationResult(
        edges=pd.DataFrame(
            columns=[
                "source_detection_id",
                "target_detection_id",
                "source_z",
                "target_z",
                "plane_offset",
                "dx_um",
                "dy_um",
                "xy_distance_um",
                "dz_um",
            ]
        ),
        membership=pd.DataFrame([{"object_id": 1, "detection_id": 1}]),
        objects=pd.DataFrame(
            [
                {
                    "object_id": 1,
                    "x": 2,
                    "y": 3,
                    "z": 0,
                    "x_float": 2.0,
                    "y_float": 3.0,
                    "z_float": 0.0,
                    "n_detections": 1,
                    "n_planes": 1,
                    "z_min": 0,
                    "z_max": 0,
                }
            ]
        ),
    )

    paths = save_deduplication_outputs(
        dataset,
        result,
        tmp_path,
        write_edge_table=True,
    )

    assert paths.objects_table.suffix == ".parquet"
    assert paths.membership_table.suffix == ".parquet"
    assert paths.edges_table is not None
    assert paths.edges_table.suffix == ".parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(paths.objects_table), result.objects)
    pd.testing.assert_frame_equal(pd.read_parquet(paths.membership_table), result.membership)


def test_region_assignments_use_parquet_and_summary_remains_csv(tmp_path: Path) -> None:
    assigned = pd.DataFrame(
        [
            {
                "object_id": 1,
                "x": 2,
                "y": 3,
                "z": 0,
                "region_id": 10,
                "assignment_status": "assigned",
            }
        ]
    )
    summary = pd.DataFrame([{"region_id": 10, "object_count": 1}])

    paths = save_instance_region_quantification_outputs(
        assigned,
        summary,
        _metadata(),
        tmp_path,
        source_name="subject_1_objects.parquet",
        annotation=LabelVolume(
            path=tmp_path / "annotation.nii.gz",
            data=np.zeros(_metadata().space.shape, dtype=np.uint16),
            space=_metadata().space,
        ),
    )

    assert paths.assigned_objects_table.suffix == ".parquet"
    assert paths.region_summary_csv.suffix == ".csv"
    assert paths.qc_png is not None
    assert paths.qc_png.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(paths.assigned_objects_table), assigned)
    pd.testing.assert_frame_equal(pd.read_csv(paths.region_summary_csv), summary)
