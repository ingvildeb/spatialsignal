# Changelog

## [0.2.1] - 2026-11-13

### Added

- Optional stable `artifact_id` values on SpatialSignal dataset metadata.
- Optional typed `processing.inputs` references for SpatialSignal or external
  upstream artifacts.
- A v0.2 metadata contract and read-only audit of existing project sidecars.

### Compatibility

- Schema-v0.1 sidecars and schema-v0.2 sidecars without formal lineage remain
  readable without mutation or migration.
- Existing dataset writers continue to emit their established no-lineage
  metadata unless a caller explicitly supplies v0.2 identity and input
  references. Automatic ID generation is intentionally deferred.
- Corrected `spatialsignal.__version__` to match the released package version.

## [0.2.0] - 2026-11-03

### Added

- Hemisphere-aware atlas-region quantification.
- Registration-aware point transformation and reference-space voxelization.
- Calibrated morphology filtering for instance detections.
- Materialization of matched cross-channel object components as unique
  colocalized populations with explicit provenance.

### Changed

- Cross-plane deduplication semantics and validation were tightened for routine
  project workflows.
- The registration extra now uses the tagged `atlasspace v0.2.3` release.

