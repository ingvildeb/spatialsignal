"""Acceptance tests for optional SpatialSignal metadata v0.2 identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatialsignal.models import (
    ArtifactReference,
    DataRepresentation,
    DatasetMetadata,
    ProcessingProvenance,
    SpaceDefinition,
)


def _space() -> SpaceDefinition:
    return SpaceDefinition(
        space_name="subject_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[10, 20, 30],
        resolution_um=[1.8, 1.8, 5.0],
    )


def _representation() -> DataRepresentation:
    return DataRepresentation(
        kind="point_cloud",
        representation_type="cleaned_objects",
    )


def test_v02_artifact_identity_and_inputs_round_trip(tmp_path: Path) -> None:
    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.2.0",
        artifact_id="objects-artifact",
        space=_space(),
        representation=_representation(),
        processing=ProcessingProvenance(
            stage="deduplicate_across_planes",
            inputs=[
                ArtifactReference(
                    role="source_pointcloud",
                    artifact_id="pointcloud-artifact",
                    schema_name="spatialsignal.dataset_metadata",
                    name="subject_pointcloud.parquet",
                )
            ],
            parameters={"max_xy_distance_um": 3.0},
        ),
    )
    path = tmp_path / "subject_objects_space.json"

    metadata.to_json(path)
    loaded = DatasetMetadata.from_json(path)

    assert loaded == metadata
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_id"] == "objects-artifact"
    assert payload["processing"]["inputs"] == [
        {
            "role": "source_pointcloud",
            "artifact_id": "pointcloud-artifact",
            "schema_name": "spatialsignal.dataset_metadata",
            "name": "subject_pointcloud.parquet",
        }
    ]


@pytest.mark.parametrize(
    ("role", "artifact_id", "schema_name", "name"),
    (
        (
            "registration",
            "registration-artifact",
            "atlasspace.registration_result",
            "registration_result.json",
        ),
        (
            "segmentation_set",
            "cellpose-run-artifact",
            "lsfm.cellpose_run",
            "cellpose_run.json",
        ),
    ),
)
def test_external_input_reference_is_generic(
    role: str, artifact_id: str, schema_name: str, name: str
) -> None:
    reference = ArtifactReference(
        role=role,
        artifact_id=artifact_id,
        schema_name=schema_name,
        name=name,
    )

    assert ArtifactReference.from_dict(reference.to_dict()) == reference


def test_v01_metadata_without_identity_remains_unchanged(tmp_path: Path) -> None:
    payload = {
        "schema_name": "spatialsignal.dataset_metadata",
        "schema_version": "0.1.0",
        "space": _space().to_dict(),
        "representation": _representation().to_dict(),
        "processing": {
            "stage": "legacy_stage",
            "parameters": {"legacy_project_parameter": True},
        },
    }
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    metadata = DatasetMetadata.from_json(source)

    assert metadata.artifact_id is None
    assert metadata.processing is not None
    assert metadata.processing.inputs is None
    assert metadata.to_dict() == payload


def test_v02_identity_and_inputs_are_independently_optional() -> None:
    without_lineage = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.2.0",
        space=_space(),
        representation=_representation(),
    )
    identity_only = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.2.0",
        artifact_id="adopted-artifact-without-lineage",
        space=_space(),
        representation=_representation(),
    )

    assert "artifact_id" not in without_lineage.to_dict()
    assert "processing" not in without_lineage.to_dict()
    assert identity_only.to_dict()["artifact_id"] == "adopted-artifact-without-lineage"


@pytest.mark.parametrize(
    ("field", "value"),
    (("role", ""), ("artifact_id", ""), ("schema_name", ""), ("name", "")),
)
def test_artifact_reference_rejects_empty_identifiers(
    field: str, value: str
) -> None:
    values = {
        "role": "source",
        "artifact_id": "upstream-artifact",
        "schema_name": "external.schema",
        "name": "source.json",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ArtifactReference(**values)


def test_dataset_metadata_rejects_empty_artifact_id() -> None:
    with pytest.raises(ValueError, match="artifact_id"):
        DatasetMetadata(
            schema_name="spatialsignal.dataset_metadata",
            schema_version="0.2.0",
            artifact_id="",
            space=_space(),
            representation=_representation(),
        )

