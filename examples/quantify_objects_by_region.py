"""Example runner for assigning cleaned objects to atlas regions in subject space."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spatialsignal.io import save_instance_region_quantification_outputs
from spatialsignal.integration import (
    load_atlasspace_registration_folder,
    load_registration_annotation_volume,
    load_registration_brain_mask_volume,
)
from spatialsignal.models import DatasetMetadata, PointCloudDataset
from spatialsignal.quantification import (
    assign_objects_to_regions,
    enrich_region_summary_with_atlas,
    remap_pointcloud_dataset_to_space,
    summarize_objects_by_region,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for subject-space region quantification."""

    parser = argparse.ArgumentParser(
        description=(
            "Assign cleaned object centroids to subject-space atlas regions using a "
            "warped annotation from an atlasspace registration folder."
        )
    )
    parser.add_argument(
        "--objects-table",
        required=True,
        help="Path to a cleaned object table such as {subject}_objects.parquet.",
    )
    parser.add_argument(
        "--space-json",
        required=True,
        help="Path to the matching objects metadata JSON such as {subject}_objects_space.json.",
    )
    parser.add_argument(
        "--registration-dir",
        required=True,
        help="Path to an atlasspace registration output folder.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where assigned objects and region summaries should be written.",
    )
    parser.add_argument(
        "--subject-name",
        default=None,
        help="Optional subject name override. If omitted, inferred from the objects table stem.",
    )
    parser.add_argument(
        "--annotation-name",
        default="annotation",
        help="Transformed segmentation name to use as the annotation volume.",
    )
    parser.add_argument(
        "--brain-mask-name",
        default="brain_mask",
        help="Optional transformed segmentation name to use as the brain mask.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include annotation label 0 in the per-region summary output.",
    )
    parser.add_argument(
        "--ontology-preset",
        default="allen_ccfv3",
        help="atlaslevels ontology preset used to name regions.",
    )
    parser.add_argument(
        "--region-id-space",
        choices=("allen", "kimlab16bit"),
        default="kimlab16bit",
        help="ID namespace stored in the transformed annotation volume.",
    )
    return parser.parse_args()


def main() -> int:
    """Assign cleaned objects to subject-space regions and write summaries."""

    args = parse_args()
    objects_table = Path(args.objects_table)
    metadata = DatasetMetadata.from_json(Path(args.space_json))
    objects = pd.read_parquet(objects_table)
    subject_name = args.subject_name or infer_subject_name_from_objects_path(objects_table)

    objects_dataset = PointCloudDataset(
        subject_name=subject_name,
        points=objects,
        metadata=metadata,
    )
    objects_dataset.validate_spatial_points()

    registration = load_atlasspace_registration_folder(
        Path(args.registration_dir),
        annotation_name=args.annotation_name,
        brain_mask_name=args.brain_mask_name,
    )
    annotation_volume = load_registration_annotation_volume(registration)
    brain_mask_volume = load_registration_brain_mask_volume(registration)
    remapped_dataset = remap_pointcloud_dataset_to_space(
        objects_dataset,
        annotation_volume.space,
        source_name=objects_table.name,
        parameters={
            "registration_dir": str(args.registration_dir),
            "annotation_name": args.annotation_name,
            "brain_mask_name": args.brain_mask_name,
        },
    )

    assigned_objects = assign_objects_to_regions(
        remapped_dataset.points,
        remapped_dataset.space,
        annotation_volume.data,
        annotation_space=annotation_volume.space,
        brain_mask_data=(brain_mask_volume.data if brain_mask_volume is not None else None),
    )
    region_summary = summarize_objects_by_region(
        assigned_objects,
        remapped_dataset.space,
        annotation_volume.data,
        annotation_space=annotation_volume.space,
        area_measurement_space=objects_dataset.space,
        include_background=args.include_background,
    )
    region_summary = enrich_region_summary_with_atlas(
        region_summary,
        ontology_preset=args.ontology_preset,
        region_id_space=args.region_id_space,
    )

    output_paths = save_instance_region_quantification_outputs(
        assigned_objects,
        region_summary,
        remapped_dataset.metadata,
        Path(args.out_dir),
        source_name=objects_table.name,
        parameters={
            "registration_dir": str(args.registration_dir),
            "annotation_name": args.annotation_name,
            "brain_mask_name": args.brain_mask_name,
            "include_background": args.include_background,
            "ontology_preset": args.ontology_preset,
            "region_id_space": args.region_id_space,
            "source_space_name": objects_dataset.space.space_name,
            "source_orientation": objects_dataset.space.orientation,
            "source_resolution_um": list(objects_dataset.space.resolution_um),
            "source_shape": list(objects_dataset.space.shape),
            "target_space_name": annotation_volume.space.space_name,
            "target_orientation": annotation_volume.space.orientation,
            "target_resolution_um": list(annotation_volume.space.resolution_um),
            "target_shape": list(annotation_volume.space.shape),
        },
    )

    print("Subject-space region quantification complete")
    print(f"  subject_name: {subject_name}")
    print(f"  registration_dir: {args.registration_dir}")
    print(f"  annotation_path: {registration.annotation_path}")
    if registration.brain_mask_path is not None:
        print(f"  brain_mask_path: {registration.brain_mask_path}")
    print(f"  assigned_objects_table: {output_paths.assigned_objects_table}")
    print(f"  assigned_objects_json: {output_paths.assigned_objects_json}")
    print(f"  region_summary_csv: {output_paths.region_summary_csv}")
    return 0


def infer_subject_name_from_objects_path(table_path: Path) -> str:
    """Infer the subject name from a cleaned object table path."""

    suffixes = ("_objects", "_pointcloud")
    stem = table_path.stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


if __name__ == "__main__":
    raise SystemExit(main())
