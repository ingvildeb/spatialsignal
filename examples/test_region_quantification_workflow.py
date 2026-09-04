from pathlib import Path

import pandas as pd

from spatialsignal.io import save_instance_region_quantification_outputs
from spatialsignal.integration import (
    load_atlasspace_registration_folder,
    load_label_volume,
)
from spatialsignal.models import DatasetMetadata, PointCloudDataset
from spatialsignal.quantification import quantify_objects_by_region

OBJECTS_TABLE = Path(
    r"Z:\LSFM\2026\2026_03\2026_03_16\20260316_10_38_12_NB_101362_F_P14_B6NJ_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\test_spatialsignal\101362_objects.parquet"
)
SPACE_JSON = Path(
    r"Z:\LSFM\2026\2026_03\2026_03_16\20260316_10_38_12_NB_101362_F_P14_B6NJ_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\test_spatialsignal\101362_objects_space.json"
)
REGISTRATION_DIR = Path(
    r"Z:\LSFM\2026\2026_03\2026_03_16\20260316_10_38_12_NB_101362_F_P14_B6NJ_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\registration"
)
OUT_DIR = Path(
    r"Z:\LSFM\2026\2026_03\2026_03_16\20260316_10_38_12_NB_101362_F_P14_B6NJ_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\test_spatialsignal"
)
SUBJECT_NAME: str | None = "101362"
ANNOTATION_NAME = "annotation"
INCLUDE_BACKGROUND_REGION = False
ONTOLOGY_PRESET = "allen_ccfv3"
REGION_ID_SPACE = "kimlab16bit"


def infer_subject_name_from_objects_path(table_path: Path) -> str:
    """Infer the subject name from a cleaned object table path."""

    suffixes = ("_objects", "_pointcloud")
    stem = table_path.stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


if __name__ == "__main__":
    metadata = DatasetMetadata.from_json(SPACE_JSON)
    objects = pd.read_parquet(OBJECTS_TABLE)
    subject_name = SUBJECT_NAME or infer_subject_name_from_objects_path(OBJECTS_TABLE)

    objects_dataset = PointCloudDataset(
        subject_name=subject_name,
        points=objects,
        metadata=metadata,
    )
    objects_dataset.validate_spatial_points()

    registration = load_atlasspace_registration_folder(REGISTRATION_DIR)
    annotation_path = registration.transformed_segmentations[ANNOTATION_NAME]
    annotation_volume = load_label_volume(
        annotation_path,
    )
    result = quantify_objects_by_region(
        objects_dataset,
        annotation_volume,
        ontology_preset=ONTOLOGY_PRESET,
        region_id_space=REGION_ID_SPACE,
        include_background=INCLUDE_BACKGROUND_REGION,
    )

    output_paths = save_instance_region_quantification_outputs(
        result.assigned_objects,
        result.region_summary,
        result.remapped_objects.metadata,
        OUT_DIR,
        source_name=OBJECTS_TABLE.name,
        annotation=annotation_volume,
        parameters={
            "registration_dir": str(REGISTRATION_DIR),
            "annotation_name": ANNOTATION_NAME,
            "include_background_region": INCLUDE_BACKGROUND_REGION,
            "ontology_preset": ONTOLOGY_PRESET,
            "region_id_space": REGION_ID_SPACE,
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
    print(f"  annotation_path: {annotation_path}")
    print(f"  assigned_objects_table: {output_paths.assigned_objects_table}")
    print(f"  assigned_objects_json: {output_paths.assigned_objects_json}")
    print(f"  region_summary_csv: {output_paths.region_summary_csv}")
    print(f"  quantification_qc_png: {output_paths.qc_png}")
    print(f"  region_summary_csv: {output_paths.region_summary_csv}")
