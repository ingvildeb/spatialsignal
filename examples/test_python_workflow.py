from pathlib import Path

from spatialsignal.io import (
    make_subject_output_stem,
    save_deduplication_outputs,
    save_instance_region_quantification_outputs,
    save_voxel_map_outputs,
)
from spatialsignal.integration import (
    load_atlasspace_registration_folder,
    load_registration_annotation_volume,
    load_registration_brain_mask_volume,
)
from spatialsignal.models import PointCloudDataset
from spatialsignal.pointcloud import (
    build_pointcloud_from_masks,
    deduplicate_across_planes,
)
from spatialsignal.quantification import (
    assign_objects_to_regions,
    enrich_region_summary_with_atlas,
    remap_pointcloud_dataset_to_space,
    summarize_objects_by_region,
)
from spatialsignal.voxelization import (
    count_map_to_density_map,
    make_subject_analysis_space,
    voxelize_to_space,
)

MASK_DIR = Path(
    r"Z:\LSFM\2026\2026_03\2026_03_16\20260316_10_38_12_NB_101362_F_P14_B6NJ_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\_02_ml_result\ch2"
)
OUT_DIR = Path(
    r"Z:\LSFM\2026\2026_03\2026_03_16\20260316_10_38_12_NB_101362_F_P14_B6NJ_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\test_spatialsignal"
)
SUBJECT_NAME = "101362"
SPACE_NAME = "subject_space"
ORIENTATION = "las"
RESOLUTION_UM = [1.8, 1.8, 20.0]
INDEXING = "zero_based"
MAX_WORKERS = 1
MAX_PLANE_OFFSET = 1
MAX_XY_DISTANCE_UM = 3.0
MAX_N_PLANES = 2
ANALYSIS_RESOLUTION_UM = [20.0, 20.0, 20.0]
ANALYSIS_SPACE_NAME = "subject_analysis_space"
REGISTRATION_DIR: Path | None = None
INCLUDE_BACKGROUND_REGION = False
ONTOLOGY_PRESET = "allen_ccfv3"
REGION_ID_SPACE = "kimlab16bit"


if __name__ == "__main__":
    output_stem = make_subject_output_stem(SUBJECT_NAME)

    build_pointcloud_from_masks(
        mask_dir=MASK_DIR,
        out_dir=OUT_DIR,
        subject_name=SUBJECT_NAME,
        space_name=SPACE_NAME,
        orientation=ORIENTATION,
        resolution_um=RESOLUTION_UM,
        indexing=INDEXING,
        max_workers=MAX_WORKERS,
    )

    dataset = PointCloudDataset.from_files(
        table_path=OUT_DIR / f"{SUBJECT_NAME}_pointcloud.parquet",
        json_path=OUT_DIR / f"{SUBJECT_NAME}_pointcloud_space.json",
    )
    dataset.validate()

    result = deduplicate_across_planes(
        dataset.points,
        dataset.space,
        max_plane_offset=MAX_PLANE_OFFSET,
        max_xy_distance_um=MAX_XY_DISTANCE_UM,
        max_n_planes=MAX_N_PLANES,
    )

    dedup_paths = save_deduplication_outputs(
        dataset,
        result,
        OUT_DIR,
        source_name=f"{output_stem}_pointcloud.parquet",
        parameters={
            "max_plane_offset": MAX_PLANE_OFFSET,
            "max_xy_distance_um": MAX_XY_DISTANCE_UM,
            "max_n_planes": MAX_N_PLANES,
        },
        output_stem=output_stem,
        write_edge_table=True,
    )
    objects_dataset = PointCloudDataset.from_files(
        dedup_paths.objects_table,
        dedup_paths.objects_json,
        subject_name=SUBJECT_NAME,
    )
    objects_dataset.validate_spatial_points()
    analysis_space = make_subject_analysis_space(
        objects_dataset.space,
        analysis_resolution_um=ANALYSIS_RESOLUTION_UM,
        space_name=ANALYSIS_SPACE_NAME,
    )
    count_map = voxelize_to_space(objects_dataset, analysis_space)
    density_map = count_map_to_density_map(count_map)
    count_paths = save_voxel_map_outputs(
        count_map,
        OUT_DIR,
        name_suffix="count_map",
        output_stem=output_stem,
    )
    density_paths = save_voxel_map_outputs(
        density_map,
        OUT_DIR,
        name_suffix="density_map",
        output_stem=output_stem,
    )

    print(len(result.objects))
    print(count_map.data.shape)
    print(dedup_paths.objects_table)
    print(dedup_paths.objects_json)
    print(dedup_paths.membership_table)
    if dedup_paths.edges_table is not None:
        print(dedup_paths.edges_table)
    print(count_paths.array_path)
    print(count_paths.metadata_path)
    if count_paths.nifti_written:
        print(count_paths.nifti_path)
    print(density_paths.array_path)
    print(density_paths.metadata_path)
    if density_paths.nifti_written:
        print(density_paths.nifti_path)

    if REGISTRATION_DIR is not None:
        registration = load_atlasspace_registration_folder(REGISTRATION_DIR)
        annotation_volume = load_registration_annotation_volume(registration)
        brain_mask_volume = load_registration_brain_mask_volume(registration)

        remapped_dataset = remap_pointcloud_dataset_to_space(
            objects_dataset,
            annotation_volume.space,
            source_name=dedup_paths.objects_table.name,
            parameters={
                "registration_dir": str(REGISTRATION_DIR),
                "annotation_path": str(registration.annotation_path),
                "brain_mask_path": (
                    str(registration.brain_mask_path)
                    if registration.brain_mask_path is not None
                    else None
                ),
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
            include_background=INCLUDE_BACKGROUND_REGION,
        )
        region_summary = enrich_region_summary_with_atlas(
            region_summary,
            ontology_preset=ONTOLOGY_PRESET,
            region_id_space=REGION_ID_SPACE,
        )
        quant_paths = save_instance_region_quantification_outputs(
            assigned_objects,
            region_summary,
            remapped_dataset.metadata,
            OUT_DIR,
            source_name=dedup_paths.objects_table.name,
            parameters={
                "registration_dir": str(REGISTRATION_DIR),
                "annotation_path": str(registration.annotation_path),
                "brain_mask_path": (
                    str(registration.brain_mask_path)
                    if registration.brain_mask_path is not None
                    else None
                ),
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
            output_stem=output_stem,
        )
        print(quant_paths.assigned_objects_table)
        print(quant_paths.assigned_objects_json)
        print(quant_paths.region_summary_csv)
