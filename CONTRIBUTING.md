# Contributing to spatialsignal

Contributions, bug reports, and questions are welcome. Please open a GitHub
issue before starting a substantial change so that its scope and data contract
can be discussed first.

## Development setup

Create and activate a Python 3.10 or newer virtual environment, then install the
package and development tools from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite and build the distribution artifacts:

```bash
python -m pytest -q
python -m build
```

You can also exercise the main instance-segmentation path without external data:

```bash
python examples/quickstart.py --out-dir quickstart-output
```

## Pull requests

- Keep reusable spatial operations in `src/spatialsignal/`; project-specific
  orchestration belongs in the consuming project.
- Add or update tests whenever behavior or a data contract changes.
- Update the README or relevant document under `docs/` when public behavior
  changes.
- Preserve explicit coordinate, indexing, representation, and provenance
  metadata in every spatial output.
- Use synthetic or appropriately shareable fixtures. Do not commit private data,
  credentials, internal storage paths, or identifying subject metadata.

`spatialsignal` is pre-1.0 software. Backward compatibility still matters, but a
change to an evolving API may be accepted when it is documented and justified.

## Reporting problems

Please include the Python version, operating system, package version or commit,
a minimal reproducing example, and the complete error message. Do not attach
private microscopy data to a public issue.
