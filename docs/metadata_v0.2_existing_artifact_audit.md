# SpatialSignal v0.2 existing-artifact audit

## Status

This document records the read-only audit performed on 2026-11-13 before
planning any metadata migration for Glia Mapping, Chandelier Cell,
Triple-transgenic, and SING.

The audit did not create, rewrite, move, or delete project files. It inspected
current canonical SpatialSignal sidecars and checked the presence of their
expected primary Parquet or NIfTI artifacts. Archived Glia runs and
registration alternatives were intentionally excluded.

## Executive conclusion

All audited artifacts are compatible with the SpatialSignal v0.2 policy.
They are readable, structurally usable schema v0.1 artifacts and may continue
to be reused as **no lineage** without migration or recomputation.

Migration feasibility varies by artifact family:

- some derived artifacts have unambiguous direct SpatialSignal inputs and can
  receive complete direct lineage in an explicit migration;
- artifacts that depend on segmentations, annotations, hemisphere maps, or
  registrations can receive stable identities but will retain no lineage until
  those external inputs expose stable IDs; and
- colocalization evidence needs one small v0.2 schema decision before its
  lineage migration can be finalized.

No audited artifact is currently v0.2-standard because none declares an
`artifact_id` or formal `processing.inputs`.

## Audit coverage

| Project | Canonical analysis roots inspected | Relevant sidecars | Notes |
| --- | ---: | ---: | --- |
| Glia Mapping | 216 `_spatialsignal` roots | 358 | Frozen registration batch inventories, including historical batches |
| Chandelier Cell | 30 `_spatialsignal` roots | 336 | 25 subjects contained cell-quantification artifacts |
| Triple-transgenic | 23 `_spatialsignal` roots | 108 | 48 subjects were listed; 23 currently contained an analysis root |
| SING | 172 `_sing` roots | 1,666 | 218 unique lab-drive paths; 136 subjects contained relevant sidecars |
| **Total** |  | **2,468** | |

Every inspected JSON sidecar was readable. Every expected point-cloud,
object, quantified-object, and count-map primary artifact was present. All 28
colocalization evidence sets had both their detection-match and
object-relationship tables.

## Observed metadata by project

### Glia Mapping

| Artifact family | Files | Schema | Formal IDs/inputs | Historical project fields |
| --- | ---: | --- | --- | --- |
| Point clouds | 76 | 0.1.0 | None | `direct_input_signatures` in 1 |
| Deduplicated objects | 74 | 0.1.0 | None | `direct_input_signatures` in 72 |
| Quantified objects | 69 | 0.1.0 | None | `direct_input_signatures` in 67 |
| Subject count maps | 72 | 0.1.0 | None | `direct_input_signatures` in 67 |
| Reference count maps | 67 | 0.1.0 | None | `direct_input_signatures` in 67 |

The historical signatures include `mtime_ns`. They are not stable identity and
must not be translated into formal lineage. They may be retained as inert
historical metadata during migration.

### Chandelier Cell

| Artifact family | Files | Schema | Formal IDs/inputs | Historical project fields |
| --- | ---: | --- | --- | --- |
| Point clouds | 75 | 0.1.0 | None | `mask_inventory` with `mtime_ns` |
| Deduplicated objects | 75 | 0.1.0 | None | None |
| Colocalization evidence | 25 | `spatialsignal.colocalization` 0.1.0 | None | `source_signatures` with `mtime_ns` |
| Colocalized objects | 25 | 0.1.0 | None | `source_signatures` with `mtime_ns` |
| Quantified objects | 68 | 0.1.0 | None | `input_signature` with `mtime_ns` |
| Subject count maps | 68 | 0.1.0 | None | `input_signature` with `mtime_ns` |

These signature fields are project-specific historical metadata, not formal
SpatialSignal lineage.

### Triple-transgenic

| Artifact family | Files | Schema | Formal IDs/inputs | Historical project fields |
| --- | ---: | --- | --- | --- |
| Point clouds | 6 | 0.1.0 | None | None |
| Deduplicated objects | 6 | 0.1.0 | None | None |
| Colocalization evidence | 3 | `spatialsignal.colocalization` 0.1.0 | None | None |
| Colocalized objects | 3 | 0.1.0 | None | None |
| Quantified objects | 45 | 0.1.0 | None | None |
| Subject count maps | 45 | 0.1.0 | None | None |

### SING

| Artifact family | Files | Schema | Formal IDs/inputs | Historical project fields |
| --- | ---: | --- | --- | --- |
| Point clouds | 272 | 0.1.0 | None | None |
| Morphology-rejected detections | 272 | 0.1.0 | None | None |
| Deduplicated objects | 272 | 0.1.0 | None | None |
| Quantified objects | 332 | 0.1.0 | None | None |
| Subject count maps | 264 | 0.1.0 | None | None |
| Reference count maps | 254 | 0.1.0 | None | None |

## Migration classification by artifact family

The classifications below describe what an explicit, validated migration tool
could do. Nothing should migrate automatically during read or reuse.

### Identity adoption with no lineage

These artifacts can be validated, upgraded to schema v0.2, and assigned stable
`artifact_id` values, but complete direct lineage cannot currently be proven:

- **Point clouds:** their segmentation-set inputs do not yet expose stable
  artifact IDs. Some historical point clouds also contain no processing
  provenance.
- **Quantified objects:** the source object dataset is identifiable, but the
  annotation and hemisphere inputs do not expose stable IDs.
- **Subject count maps:** the source object dataset is identifiable, but the
  subject annotation grid does not expose a stable ID.
- **Reference count maps:** the source object dataset is identifiable, but the
  registration/reference-grid inputs do not expose stable IDs.

Migration may record the identifiable SpatialSignal input while omitting the
unidentified external inputs. Under the v0.2 contract, the artifact's lineage
state remains **no lineage** because not every required direct input can be
verified.

### Complete direct-lineage migration

After their upstream point clouds receive IDs, these families have
unambiguous canonical direct inputs and can receive complete direct lineage:

- **Morphology-rejected detections:** direct input is the corresponding source
  point cloud.
- **Deduplicated objects:** direct input is the corresponding source point
  cloud.

This is valid even when the point cloud itself has no historical lineage.
Lineage is evaluated one processing edge at a time, not as a claim that every
ancestor is known.

### Colocalization pending one schema decision

Colocalization evidence uses the separate
`spatialsignal.colocalization` schema and comprises:

- a colocalization metadata sidecar;
- a detection-match table; and
- an object-relationship table.

The evidence is consumed to produce the colocalized-object dataset. Before
migration, SpatialSignal should define whether this evidence set receives its
own optional `artifact_id` and `processing.inputs`, or is formally an auxiliary
component of the colocalized-object artifact.

The recommended model is to treat the evidence set as its own artifact. Its
two direct inputs are the participating channel-object datasets. Once this is
formalized, both colocalization evidence and colocalized objects are safely
migratable with complete direct lineage.

### Already v0.2-standard

None of the audited artifacts is already v0.2-standard.

## Recommended future migration order

If migration is pursued later, it should be explicit, atomic, and performed in
topological order:

1. Structurally validate every complete artifact set selected for migration.
2. Assign v0.2 identities to point clouds without inventing segmentation
   lineage.
3. Migrate morphology-rejected detections and deduplicated objects, recording
   their point-cloud inputs.
4. After formalizing the colocalization evidence schema, migrate evidence sets
   and then colocalized objects.
5. Migrate quantified objects and count maps with identities and any provable
   SpatialSignal inputs, leaving their overall state as no lineage where
   external input IDs remain unavailable.
6. Preserve project-specific timestamp/signature fields only as inert
   historical metadata. Never use them as artifact identity or staleness
   evidence.
7. Revalidate every migrated sidecar and primary artifact before accepting the
   migration.

Existing v0.1 artifacts do not need to wait for this work. They remain valid
and reusable under the no-lineage compatibility policy.

## Scope limitations

- The audit covered current canonical analysis trees selected from frozen
  project batch inventories and the SING path inventory.
- Archived Glia runs and registration-run alternatives were excluded.
- The audit checked JSON readability and expected artifact-set presence. It did
  not deeply read every coordinate row or voxel value; that is unnecessary for
  metadata classification and would be prohibitively expensive.
- No conclusion here makes migration mandatory or requires recomputation.

