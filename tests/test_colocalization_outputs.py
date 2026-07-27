import json
from pathlib import Path

import pandas as pd

from spatialsignal.io import save_colocalization_outputs
from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from spatialsignal.pointcloud import (
    map_detection_matches_to_objects,
    match_colocalized_detections,
)


def _dataset(detection_id: int, x: int) -> PointCloudDataset:
    points = pd.DataFrame(
        [
            {
                "detection_id": detection_id,
                "seg_num": detection_id,
                "x": x,
                "y": 5,
                "z": 0,
            }
        ]
    )
    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="native",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="zero_based",
            units="voxel",
            shape=[20, 10, 2],
            resolution_um=[1.0, 1.0, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )
    return PointCloudDataset("subject", points, metadata)


def test_save_colocalization_outputs_writes_relational_outputs_and_qc(
    tmp_path: Path,
) -> None:
    dataset_a = _dataset(1, 5)
    dataset_b = _dataset(11, 6)
    result = match_colocalized_detections(
        dataset_a,
        dataset_b,
        max_xy_distance_um=2.0,
    )
    relationships = map_detection_matches_to_objects(
        result.matches,
        pd.DataFrame([{"object_id": 100, "detection_id": 1}]),
        pd.DataFrame([{"object_id": 200, "detection_id": 11}]),
    )

    paths = save_colocalization_outputs(
        dataset_a,
        dataset_b,
        result,
        tmp_path,
        channel_a_name="GFP",
        channel_b_name="tdTomato",
        max_xy_distance_um=2.0,
        object_relationships=relationships,
        source_a="a.parquet",
        source_b="b.parquet",
    )

    assert paths.matches_table.exists()
    assert paths.relationships_table is not None
    assert paths.relationships_table.exists()
    assert paths.metadata_path.exists()
    assert paths.qc_report is not None and paths.qc_report.exists()
    assert paths.qc_images_dir is not None
    assert len(list(paths.qc_images_dir.glob("*.png"))) == 2
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["channels"]["a"]["name"] == "GFP"
    assert metadata["processing"]["summary"]["accepted_detection_matches"] == 1
    saved_relationships = pd.read_parquet(paths.relationships_table)
    assert saved_relationships.loc[0, "relationship_status"] == "one_to_one"
