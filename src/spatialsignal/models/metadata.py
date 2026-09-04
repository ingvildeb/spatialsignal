"""Shared metadata model definitions for spatial datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from spatialsignal.utils.images import read_2d_mask


@dataclass
class SpaceDefinition:
    """Metadata describing a spatial/grid definition independent of data type."""

    space_name: str
    orientation: str
    axis_labels: list[str]
    indexing: str
    units: str
    shape: list[int]
    resolution_um: list[float]
    affine_ras_mm: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the spatial definition to a plain dictionary."""

        data = {
            "space_name": self.space_name,
            "orientation": self.orientation,
            "axis_labels": self.axis_labels,
            "indexing": self.indexing,
            "units": self.units,
            "shape": self.shape,
            "resolution_um": self.resolution_um,
        }
        if self.affine_ras_mm is not None:
            data["affine_ras_mm"] = self.affine_ras_mm
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpaceDefinition":
        """Construct a spatial definition from a dictionary."""

        return cls(
            space_name=data["space_name"],
            orientation=data["orientation"],
            axis_labels=data["axis_labels"],
            indexing=data["indexing"],
            units=data["units"],
            shape=data["shape"],
            resolution_um=data["resolution_um"],
            affine_ras_mm=data.get("affine_ras_mm"),
        )

    @classmethod
    def from_mask_files(
        cls,
        *,
        space_name: str,
        orientation: str,
        resolution_um: list[float],
        indexing: str,
        mask_files: list[Path],
        axis_labels: list[str] | None = None,
    ) -> "SpaceDefinition":
        """Build a spatial definition from a stack of mask files in image voxel space."""

        if axis_labels is None:
            axis_labels = ["x", "y", "z"]

        if len(mask_files) == 0:
            raise ValueError("Found no mask files, cannot build space definition")
        if len(resolution_um) != 3:
            raise ValueError(f"resolution_um must have length 3, got {resolution_um}")
        if len(axis_labels) != 3:
            raise ValueError(f"axis_labels must have length 3, got {axis_labels}")

        first_mask = read_2d_mask(mask_files[0])
        shape = [int(first_mask.shape[1]), int(first_mask.shape[0]), len(mask_files)]

        return cls(
            space_name=space_name,
            orientation=orientation,
            axis_labels=axis_labels,
            indexing=indexing,
            units="voxel",
            shape=shape,
            resolution_um=[float(value) for value in resolution_um],
        )


@dataclass
class DataRepresentation:
    """Semantic description of what kind of spatial dataset is being represented."""

    kind: str
    representation_type: str
    value_units: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the data representation to a plain dictionary."""

        data = {
            "kind": self.kind,
            "representation_type": self.representation_type,
        }
        if self.value_units is not None:
            data["value_units"] = self.value_units
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataRepresentation":
        """Construct a data representation from a dictionary."""

        return cls(
            kind=data["kind"],
            representation_type=data["representation_type"],
            value_units=data.get("value_units"),
        )


@dataclass
class ProcessingProvenance:
    """Provenance describing how a spatial dataset was produced."""

    stage: str
    source_name: str | None = None
    parameters: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance to a plain dictionary."""

        data: dict[str, Any] = {"stage": self.stage}
        if self.source_name is not None:
            data["source_name"] = self.source_name
        if self.parameters is not None:
            data["parameters"] = self.parameters
        if self.summary is not None:
            data["summary"] = self.summary
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingProvenance":
        """Construct provenance from a dictionary."""

        return cls(
            stage=data["stage"],
            source_name=data.get("source_name"),
            parameters=data.get("parameters"),
            summary=data.get("summary"),
        )


@dataclass
class DatasetMetadata:
    """Metadata for a spatial dataset, gathered from reusable metadata pieces."""

    schema_name: str
    schema_version: str
    space: SpaceDefinition
    representation: DataRepresentation
    processing: ProcessingProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset metadata to the JSON sidecar structure."""

        data: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "space": self.space.to_dict(),
            "representation": self.representation.to_dict(),
        }
        if self.processing is not None:
            data["processing"] = self.processing.to_dict()
        return data

    def to_json(self, output_path: Path) -> None:
        """Write dataset metadata to JSON."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        """Construct dataset metadata from the canonical nested dictionary layout."""

        return cls(
            schema_name=data["schema_name"],
            schema_version=data["schema_version"],
            space=SpaceDefinition.from_dict(data["space"]),
            representation=DataRepresentation.from_dict(data["representation"]),
            processing=(
                ProcessingProvenance.from_dict(data["processing"])
                if "processing" in data
                else None
            ),
        )

    @classmethod
    def from_json(cls, json_path: Path) -> "DatasetMetadata":
        """Load dataset metadata from a JSON file."""

        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def from_mask_files(
        cls,
        *,
        space_name: str,
        orientation: str,
        resolution_um: list[float],
        indexing: str,
        mask_files: list[Path],
        representation_type: str = "point_centroids",
        axis_labels: list[str] | None = None,
        schema_name: str = "spatialsignal.dataset_metadata",
        schema_version: str = "0.1.0",
    ) -> "DatasetMetadata":
        """Build dataset metadata from a stack of mask files in image voxel space."""

        space = SpaceDefinition.from_mask_files(
            space_name=space_name,
            orientation=orientation,
            resolution_um=resolution_um,
            indexing=indexing,
            mask_files=mask_files,
            axis_labels=axis_labels,
        )
        return cls(
            schema_name=schema_name,
            schema_version=schema_version,
            space=space,
            representation=DataRepresentation(
                kind="point_cloud",
                representation_type=representation_type,
            ),
        )


def build_dataset_metadata(
    *,
    space_name: str,
    orientation: str,
    resolution_um: list[float],
    indexing: str,
    mask_files: list[Path],
    representation_type: str = "point_centroids",
    axis_labels: list[str] | None = None,
    schema_name: str = "spatialsignal.dataset_metadata",
    schema_version: str = "0.1.0",
) -> dict[str, Any]:
    """Build dataset metadata for a spatial dataset in image voxel space."""

    return DatasetMetadata.from_mask_files(
        space_name=space_name,
        orientation=orientation,
        resolution_um=resolution_um,
        indexing=indexing,
        mask_files=mask_files,
        representation_type=representation_type,
        axis_labels=axis_labels,
        schema_name=schema_name,
        schema_version=schema_version,
    ).to_dict()


def write_dataset_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Write dataset metadata to JSON."""

    DatasetMetadata.from_dict(metadata).to_json(output_path)
