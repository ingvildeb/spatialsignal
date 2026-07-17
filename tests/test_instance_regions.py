"""Tests for instance-based region quantification."""

import numpy as np
import pandas as pd

from spatialsignal.models import SpaceDefinition
from spatialsignal.quantification import (
    enrich_region_summary_with_atlas,
    summarize_objects_by_hierarchy_level,
    summarize_objects_by_region,
)


def test_region_summary_reports_median_object_area_in_square_microns() -> None:
    registration_space = SpaceDefinition(
        space_name="registration",
        orientation="lsp",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 1, 1],
        resolution_um=[20.0, 20.0, 20.0],
    )
    native_space = SpaceDefinition(
        space_name="native",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[20, 10, 1],
        resolution_um=[1.5, 2.0, 20.0],
    )
    assigned_objects = pd.DataFrame(
        {
            "x": [0, 0, 1],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "region_id": [1, 1, 2],
            "mean_area_px": [10.0, 20.0, 100.0],
        }
    )
    annotation = np.array([[[1]], [[2]]], dtype=np.int32)

    summary = summarize_objects_by_region(
        assigned_objects,
        registration_space,
        annotation,
        annotation_space=registration_space,
        area_measurement_space=native_space,
    )

    region_one = summary.loc[summary["region_id"] == 1].iloc[0]
    assert region_one["median_object_area_um2"] == 45.0
    assert "mean_object_area_px" not in summary.columns
    assert "sum_mean_area_px" not in summary.columns


def test_region_summary_reports_median_object_eccentricity_only() -> None:
    space = SpaceDefinition(
        space_name="subject",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 1, 1],
        resolution_um=[1.5, 1.5, 20.0],
    )
    assigned_objects = pd.DataFrame(
        {
            "x": [0, 0, 1],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "region_id": [1, 1, 2],
            "mean_eccentricity": [0.2, 0.6, 0.9],
        }
    )
    annotation = np.array([[[1]], [[2]]], dtype=np.int32)

    summary = summarize_objects_by_region(assigned_objects, space, annotation)

    region_one = summary.loc[summary["region_id"] == 1].iloc[0]
    assert region_one["median_object_eccentricity"] == 0.4
    assert "mean_object_eccentricity" not in summary.columns
    assert "median_major_axis_length_um" not in summary.columns
    assert "median_minor_axis_length_um" not in summary.columns


def test_atlas_enrichment_converts_kimlab_ids_and_labels_special_rows() -> None:
    summary = pd.DataFrame(
        {
            "region_id": [20040, 0, -1],
            "object_count": [5, 1, 1],
        }
    )

    enriched = enrich_region_summary_with_atlas(
        summary,
        ontology_preset="allen_ccfv3",
        region_id_space="kimlab16bit",
    )

    atlas_row = enriched.loc[enriched["region_id"] == 20040].iloc[0]
    assert atlas_row["allen_region_id"] == 526157192
    assert atlas_row["region_acronym"] == "FRP5"
    assert atlas_row["region_name"] == "Frontal pole, layer 5"
    assert atlas_row["region_color"] == "#268F45"
    assert enriched.loc[enriched["region_id"] == 0, "region_name"].iloc[0] == "Background"
    assert (
        enriched.loc[enriched["region_id"] == -1, "region_name"].iloc[0]
        == "Out of bounds"
    )


def test_hierarchy_summary_recomputes_parent_metrics_from_objects_and_volume() -> None:
    space = SpaceDefinition(
        space_name="subject",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[3, 1, 1],
        resolution_um=[10.0, 10.0, 10.0],
    )
    assigned_objects = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [0, 0, 0, 0],
            "z": [0, 0, 0, 0],
            "region_id": [68, 667, 184, -1],
            "n_detections": [1, 2, 1, 1],
            "mean_area_px": [1.0, 9.0, 25.0, 100.0],
            "mean_eccentricity": [0.1, 0.5, 0.9, 1.0],
        }
    )
    annotation = np.array([[[68]], [[667]], [[184]]], dtype=np.int64)

    summary = summarize_objects_by_hierarchy_level(
        assigned_objects,
        space,
        annotation,
        hierarchy_preset="allen_gm_tiny",
        hierarchy_level="allen_gm_tiny_level_2",
        region_id_space="allen",
    )

    parent = summary.loc[summary["region_id"] == 184].iloc[0]
    assert parent["hierarchy_preset"] == "allen_gm_tiny"
    assert parent["is_parent_residual"] == False
    assert "hierarchy_level" not in summary.columns
    assert "source_region_id_space" not in summary.columns
    assert "allen_region_id" not in summary.columns
    assert parent["region_name"] == "Frontal pole, cerebral cortex"
    assert parent["region_voxels"] == 3
    assert parent["region_volume_mm3"] == 3e-6
    assert parent["object_count"] == 3
    assert parent["object_density_per_mm3"] == 1_000_000.0
    assert parent["detection_count"] == 4
    assert parent["median_object_area_um2"] == 900.0
    assert parent["median_object_eccentricity"] == 0.5

    out_of_bounds = summary.loc[summary["region_id"] == -1].iloc[0]
    assert out_of_bounds["region_name"] == "Out of bounds"
    assert out_of_bounds["object_count"] == 1


def test_hierarchy_summary_converts_kimlab_ids_before_parent_mapping() -> None:
    space = SpaceDefinition(
        space_name="subject",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[1, 1, 1],
        resolution_um=[20.0, 20.0, 20.0],
    )
    assigned_objects = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "z": [0],
            "region_id": [20040],
        }
    )
    annotation = np.array([[[20040]]], dtype=np.int64)

    summary = summarize_objects_by_hierarchy_level(
        assigned_objects,
        space,
        annotation,
        hierarchy_preset="allen_gm",
        hierarchy_level="CustomLevel1_gm",
        region_id_space="kimlab16bit",
    )

    row = summary.iloc[0]
    assert row["region_id"] == 184
    assert row["is_parent_residual"] == False
    assert row["region_name"] == "Frontal pole, cerebral cortex"


def test_hierarchy_summary_preserves_parent_residual_regions() -> None:
    space = SpaceDefinition(
        space_name="subject",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[1, 1, 1],
        resolution_um=[20.0, 20.0, 20.0],
    )
    assigned_objects = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "z": [0],
            "region_id": [313],
        }
    )
    annotation = np.array([[[313]]], dtype=np.int32)

    summary = summarize_objects_by_hierarchy_level(
        assigned_objects,
        space,
        annotation,
        hierarchy_preset="allen_gm",
        hierarchy_level="CustomLevel1_gm",
        region_id_space="kimlab16bit",
    )

    row = summary.iloc[0]
    assert row["region_id"] == 313
    assert row["is_parent_residual"] == True
    assert row["region_name"] == "Midbrain"
    assert row["region_voxels"] == 1
    assert row["object_count"] == 1


def test_hierarchy_summary_preserves_observed_root_label() -> None:
    space = SpaceDefinition(
        space_name="subject",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[1, 1, 1],
        resolution_um=[20.0, 20.0, 20.0],
    )
    assigned_objects = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "z": [0],
            "region_id": [997],
        }
    )
    annotation = np.array([[[997]]], dtype=np.int32)

    summary = summarize_objects_by_hierarchy_level(
        assigned_objects,
        space,
        annotation,
        hierarchy_preset="allen_gm",
        hierarchy_level="CustomLevel1_gm",
        region_id_space="kimlab16bit",
    )

    row = summary.iloc[0]
    assert row["region_id"] == 997
    assert row["is_parent_residual"] == True
    assert row["region_name"] == "root"
    assert row["region_voxels"] == 1
    assert row["object_count"] == 1
