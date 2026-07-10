from pathlib import Path

from spatialsignal.io import (
    make_subject_output_stem,
    save_deduplication_outputs,
    save_voxel_map_outputs,
)
from spatialsignal.pointcloud import (
    PointCloudDataset,
    build_pointcloud_from_masks,
    deduplicate_across_planes,
)
from spatialsignal.voxelization import make_subject_analysis_space, voxelize_to_space

MASK_DIR = Path(
    r"Z:\LSFM\2025\2025_03\2025_03_28\20250328_13_50_53_NB_IEB0014_M_P14_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\archive\_02_ml_result\ch0"
)
OUT_DIR = Path(
    r"C:\Users\SmartBrain_32C_TR\Documents\GitHub\standard_test_data\lsfm_cell_mapping\test_script"
)
SUBJECT_NAME = "IEB0014"
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
        csv_path=OUT_DIR / f"{SUBJECT_NAME}_pointcloud.csv",
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

    analysis_space = make_subject_analysis_space(
        dataset.space,
        analysis_resolution_um=ANALYSIS_RESOLUTION_UM,
        space_name=ANALYSIS_SPACE_NAME,
    )
    voxel_map = voxelize_to_space(dataset, analysis_space)

    dedup_paths = save_deduplication_outputs(
        dataset,
        result,
        OUT_DIR,
        source_name=f"{output_stem}_pointcloud.csv",
        parameters={
            "max_plane_offset": MAX_PLANE_OFFSET,
            "max_xy_distance_um": MAX_XY_DISTANCE_UM,
            "max_n_planes": MAX_N_PLANES,
        },
        output_stem=output_stem,
        write_edge_table=True,
    )
    voxel_paths = save_voxel_map_outputs(
        voxel_map,
        OUT_DIR,
        name_suffix="count_map",
        output_stem=output_stem,
    )

    print(len(result.objects))
    print(voxel_map.data.shape)
    print(dedup_paths.objects_csv)
    print(dedup_paths.objects_json)
    print(dedup_paths.membership_csv)
    if dedup_paths.edges_csv is not None:
        print(dedup_paths.edges_csv)
    print(voxel_paths.array_path)
    print(voxel_paths.metadata_path)
    if voxel_paths.nifti_written:
        print(voxel_paths.nifti_path)
