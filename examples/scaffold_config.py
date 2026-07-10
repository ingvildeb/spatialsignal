"""Example wrapper for scaffolding packaged spatialsignal starter configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.config_templates import list_templates, scaffold_template


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for config scaffolding."""

    parser = argparse.ArgumentParser(
        description="Write a packaged spatialsignal starter config to a local TOML file."
    )
    parser.add_argument(
        "--template",
        required=True,
        choices=list_templates(),
        help="Starter template identifier to scaffold.",
    )
    parser.add_argument(
        "--out",
        help="Destination TOML path. Defaults to <template>_local.toml in the current directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    return parser.parse_args()


def default_output_path(template_name: str) -> Path:
    """Return the default scaffold destination for a template."""

    return Path.cwd() / f"{template_name}_local.toml"


def main() -> int:
    """Scaffold a local config file from a packaged starter template."""

    args = parse_args()
    output_path = Path(args.out) if args.out else default_output_path(args.template)
    written_path = scaffold_template(
        args.template,
        output_path,
        overwrite=args.force,
    )
    print(f"Wrote {args.template} config to {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
