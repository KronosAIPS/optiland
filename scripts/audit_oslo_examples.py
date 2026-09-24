"""Audit a locally downloaded official OSLO demo ZIP; never execute its commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import optiland.backend as be
from optiland.fileio import load_oslo_file

SOURCE = (
    "https://lambdares.com/hubfs/Support/support/oslo/oslo_examples/OSLOLensDemos.zip"
)


def audit(archive: Path, backend: str) -> dict:
    """Return per-file permissive, strict and on-axis ray-trace outcomes."""
    be.set_backend(backend)
    rows = []
    with (
        tempfile.TemporaryDirectory(prefix="oslo-audit-") as directory,
        zipfile.ZipFile(archive) as zipped,
    ):
        for name in sorted(zipped.namelist()):
            if not name.lower().endswith(".len"):
                continue
            raw = zipped.read(name)
            # Do not extract archive-supplied paths or execute any embedded CCL.
            path = Path(directory) / "example.len"
            path.write_bytes(raw)
            row = {"file": name, "sha256": hashlib.sha256(raw).hexdigest()}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    optic = load_oslo_file(path)
                    row["import_error"] = None
                except Exception as exc:  # Audit all files, including broken examples.
                    optic = None
                    row["import_error"] = f"{type(exc).__name__}: {exc}"
                row["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    load_oslo_file(path, strict=True)
                    row["strict_error"] = None
                except Exception as exc:
                    row["strict_error"] = f"{type(exc).__name__}: {exc}"
                if optic is not None:
                    try:
                        rays = optic.trace(
                            0,
                            0,
                            optic.primary_wavelength,
                            num_rays=5,
                            distribution="line_y",
                        )
                        finite = (
                            be.isfinite(rays.x)
                            & be.isfinite(rays.y)
                            & be.isfinite(rays.z)
                        )
                        row["finite_transmitted_rays"] = int(
                            be.sum(finite & (rays.i > 0)).item()
                        )
                        row["trace_error"] = None
                    except Exception as exc:
                        row["trace_error"] = f"{type(exc).__name__}: {exc}"
            # Replace paths in strings before JSON encoding, so quotes and
            # backslashes in archive member names remain ordinary text.
            row["warnings"] = [
                message.replace(str(path), name) for message in row["warnings"]
            ]
            for key in ("import_error", "strict_error", "trace_error"):
                if row.get(key):
                    row[key] = row[key].replace(str(path), name)
            rows.append(row)
    return {
        "source": SOURCE,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "backend": backend,
        "summary": dict(
            Counter(
                "failed"
                if row["import_error"]
                else "warned"
                if row["warnings"]
                else "clean"
                for row in rows
            )
        ),
        "strict_successes": sum(row["strict_error"] is None for row in rows),
        "note": "Import and on-axis smoke success do not certify "
        "optical equivalence to OSLO.",
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    args = parser.parse_args()
    report = audit(args.archive, args.backend)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
