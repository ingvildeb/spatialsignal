# Reference-space point clouds

`transform_pointcloud_to_reference_space()` maps a native
`PointCloudDataset` into the normalized moving/template image of an AtlasSpace
registration. The native grid is first remapped to the normalized fixed image
without anatomical deformation; AtlasSpace then applies the registration in
physical space.

```python
from spatialsignal.integration import transform_pointcloud_to_reference_space
from spatialsignal.voxelization import voxelize_to_space

result = transform_pointcloud_to_reference_space(
    native_objects,
    registration_dir,
    chunk_size=250_000,
)
count_map = voxelize_to_space(
    result.dataset,
    result.dataset.space,
    processing_parameters={
        "space_kind": "reference",
        "modulation": "none",
        "point_weight": 1,
    },
)
```

The result separates non-finite and out-of-bounds transformed rows instead of
clipping them. Accepted rows preserve native coordinates under `native_*` and
retain all object, morphology, and colocalization columns. The returned target
space contains the exact normalized template NIfTI affine and shape.

These count maps are unmodulated: every accepted object contributes one count.
They must not be interpreted as Jacobian-corrected density maps. Maps on
different age-template grids must not be averaged directly.
