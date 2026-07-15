"""Tests for instance-based region quantification."""

import numpy as np
import pandas as pd

from spatialsignal.models import SpaceDefinition
from spatialsignal.quantification import (
    enrich_region_summary_with_atlas,
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
