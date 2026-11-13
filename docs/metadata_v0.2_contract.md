# SpatialSignal dataset metadata v0.2 contract

## Status and scope

This document defines the metadata contract supported by SpatialSignal's model
layer for dataset metadata schema `0.2.0`. It does not make lineage mandatory
and does not require data producers to use a particular segmentation,
registration, or workflow package. Release 0.2.1 reads and round-trips the
optional fields; established dataset writers continue to emit their existing
no-lineage metadata until automatic identity generation is implemented as a
separate change.

The contract adds two optional concepts:

- a stable identity for a logical SpatialSignal artifact set; and
- typed references to upstream artifacts used to produce a derived artifact.

Filesystem paths, file modification times, and project-specific file
signatures are not artifact identity and are outside this contract.

## Dataset artifact identity

`DatasetMetadata` may contain a top-level `artifact_id`:

```json
{
  "schema_name": "spatialsignal.dataset_metadata",
  "schema_version": "0.2.0",
  "artifact_id": "74d24d63-4cf0-4ae8-8162-75f4095a19d8",
  "space": {},
  "representation": {}
}
```

`artifact_id` identifies the logical dataset represented by the primary data
file, its SpatialSignal metadata sidecar, and any required auxiliary files.
For example, one deduplicated-object artifact may comprise the object table,
object metadata, membership table, and optional edge table.

The identifier is an opaque, non-empty string. SpatialSignal writers should
use UUIDs, but readers must not attach semantics to the identifier format.

Identity rules:

1. A newly materialized dataset receives a new `artifact_id`.
2. A recomputed or replaced dataset receives a new `artifact_id`, even when it
   is written to the same paths.
3. Copying, moving, or renaming a complete artifact set preserves its
   `artifact_id`.
4. Adding an ID to an existing artifact without formal lineage establishes the identity of
   that existing artifact; it does not establish missing historical lineage.
5. `artifact_id` is identity, not a content checksum. File presence, schema,
   shape, and domain-specific integrity remain separate validation concerns.

`artifact_id` is optional. Reading or using a dataset must not require it.

## Processing input references

`ProcessingProvenance` may contain an `inputs` list. Each entry references one
logical upstream artifact:

```json
{
  "processing": {
    "stage": "deduplicate_across_planes",
    "inputs": [
      {
        "role": "source_pointcloud",
        "artifact_id": "155dd753-e46c-488d-a16e-51914e94611b",
        "schema_name": "spatialsignal.dataset_metadata",
        "name": "IEB0078_pointcloud.parquet"
      }
    ],
    "parameters": {
      "max_plane_offset": 1,
      "max_xy_distance_um": 3.0,
      "max_n_planes": 3
    }
  }
}
```

An input reference contains:

- `role` (required): the input's role in the operation;
- `artifact_id` (required): the stable identity declared by the upstream
  artifact's own metadata contract;
- `schema_name` (optional): the upstream metadata schema that defines the ID;
- `name` (optional): a human-readable label, not an identity field.

The pair `(schema_name, artifact_id)` identifies an input when `schema_name`
is present. When it is absent, `artifact_id` alone identifies the input.

Input references are ordered only when the producing operation defines an
ordered role. Consumers should otherwise compare references by role and
identity rather than list position.

`inputs` is optional. A dataset may have:

- neither an `artifact_id` nor input references;
- its own `artifact_id` but no recorded inputs; or
- its own `artifact_id` and one or more input references.

An empty or omitted `inputs` list never means that the producer had no physical
inputs. It means only that no formal upstream artifact references were
recorded.

## External inputs

Input references are not restricted to SpatialSignal datasets. Any external
system may be referenced when its metadata declares a stable `artifact_id`.
The external system owns the meaning, lifecycle, and granularity of that ID.

For example, integrations may eventually record:

```json
{
      "role": "segmentation_set",
  "artifact_id": "cellpose-run-artifact-id",
  "schema_name": "lsfm.cellpose_run",
  "name": "cellpose_run.json"
}
```

or:

```json
{
  "role": "registration",
  "artifact_id": "registration-artifact-id",
  "schema_name": "atlasspace.registration_result",
  "name": "registration_result.json"
}
```

These examples do not create dependencies on those packages. SpatialSignal
stores and round-trips generic references; an integration or workflow layer
loads the external metadata and validates the referenced identity.

The role `segmentation_set` is intentionally independent of the producing
tool and on-disk representation. It may identify instance masks from Cellpose,
semantic segmentations, or another coherently versioned segmentation output
set. The producer-specific `schema_name` distinguishes their metadata
contracts.

When an upstream tool does not expose a stable artifact ID, the producer may
omit that input reference. It must not substitute a filesystem path,
modification timestamp, or project-specific pseudo-identity.

## Scientific parameters and lineage

`processing.parameters` remains the location for scientific and algorithmic
settings that define an operation. New schema `0.2.0` writers must not embed
artifact identity or upstream references in `parameters`.

Examples of parameters include distance thresholds, filtering rules,
coordinate rules, atlas presets, and modulation choices. Examples that do not
belong in parameters include `mtime_ns`, file inventories used as identity,
and custom keys such as `direct_input_signatures` or `input_signature`.

SpatialSignal validates the structure of metadata. A workflow layer may impose
a stronger contract by requiring particular input roles and scientific
parameters for a stage.

## Backward compatibility

SpatialSignal must continue to read schema `0.1.0` sidecars. Missing
`artifact_id` and `processing.inputs` deserialize as `None`. Reading any
sidecar is non-mutating: it must not generate an ID, change the schema version,
or rewrite the file.

Schema `0.2.0` also permits both fields to be absent. This keeps SpatialSignal
usable with manually created datasets and workflows that do not provide
formal lineage.

Existing schema `0.1.0` artifacts remain supported inputs. Routine reuse must
not require a metadata migration merely because a newer schema exists.

## Lineage validation and reuse

Lineage is evaluated for the direct inputs of one processing stage. Valid
direct lineage does not claim that every upstream ancestor has complete
lineage. An upstream artifact may have an `artifact_id` and no knowledge of
how it was originally produced; a downstream artifact can still validly
reference that upstream identity.

A workflow defines which input roles and scientific parameters are required
for its stage. Against that workflow contract, validation distinguishes:

- **valid lineage**: required IDs, input roles, and scientific parameters are
  present and match;
- **no lineage**: the dataset is structurally valid, but its own identity or
  enough required direct-input identity is unavailable to prove lineage;
- **stale lineage**: a recorded required upstream ID or scientific parameter
  differs from the actual input or requested operation; and
- **invalid metadata**: declared identity or input-reference structures are
  malformed, or the dataset fails structural validation.

| Observed artifact | Lineage state |
| --- | --- |
| Structurally valid v0.1 metadata | No lineage |
| Valid v0.2 metadata without its own ID | No lineage |
| Valid v0.2 metadata missing any required direct-input identity | No lineage |
| Valid v0.2 metadata whose required direct IDs and parameters match | Valid |
| Recorded direct ID or scientific parameter explicitly differs | Stale |
| Malformed metadata or structurally invalid artifact files | Invalid |

If some input references are present but other required roles cannot be
identified, the result is **no lineage**, not stale. A mismatch is stale only
when the recorded and actual values can both be established and differ.

No lineage is not invalid and does not by itself prohibit reuse. A workflow
may reuse a no-lineage artifact when its structural checks and available
scientific-parameter checks pass. Stale and invalid artifacts must not be
silently reused. Reuse policy belongs to the workflow layer, not to
SpatialSignal's metadata reader.

Existing project-specific signature keys must remain readable as ordinary
parameters without formal lineage. SpatialSignal must not interpret them.
New writers must not create them, while explicit migration may preserve them
as inert historical metadata rather than deleting information.

## Writer requirements

Once v0.2 writing is implemented, SpatialSignal writers should:

1. assign a new `artifact_id` to every newly materialized logical dataset;
2. assign a new ID to every derived dataset instead of inheriting its source's
   ID through `dataclasses.replace`;
3. record input references only for upstream artifacts that actually declare
   stable IDs;
4. accept omitted input references without error, including when producing a
   new identified artifact from an upstream artifact with no identity; and
5. never write filesystem modification timestamps as lineage.

## Migration boundaries

Migration is optional and explicit. Reading, validating, or consuming an
artifact must never migrate it automatically. A schema `0.1.0` artifact may
remain at `0.1.0` indefinitely.

A metadata-only migration must follow these rules:

1. Validate the complete artifact set structurally before modifying its
   sidecar. Migration must not legitimize missing, partial, or malformed data.
2. Leave primary data and auxiliary artifact files unchanged.
3. Change the sidecar schema version to `0.2.0` and assign one new
   `artifact_id` to the existing logical artifact set. Re-running migration
   must preserve that assigned ID.
4. Add an input reference only when the direct relationship is established
   from reliable existing evidence and the upstream artifact actually declares
   a stable ID.
5. Omit any historical input that cannot be established. Do not derive a
   pseudo-identity from a path, modification timestamp, file inventory, or
   content checksum.
6. Preserve scientific parameters and unrecognized historical metadata.
   Historical signature fields remain inert and are not formal lineage.
7. Write the replacement sidecar atomically. A failed migration must leave the
   original sidecar unchanged.

There is no requirement to upgrade a sidecar solely to change its version.
Migration is useful when assigning a stable identity, recording provable
direct inputs, or both.

Consequently, migration can produce either:

- a v0.2 artifact with valid formal lineage; or
- a v0.2 artifact with an ID whose historical lineage remains unavailable.

Leaving a valid v0.1 artifact untouched is a third legitimate outcome. None of
these outcomes requires recomputation.

