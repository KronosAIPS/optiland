"""The example audit distinguishes import, strict validation and tracing outcomes."""

import hashlib
import json
import runpy
import sys
import zipfile

import optiland.backend as be
from optiland.optic import Optic
from scripts import audit_oslo_examples
from scripts.audit_oslo_examples import audit


CLEAN_LENS = (
    'LEN NEW "audit" 50 3\nEBR 2\nANG 0\n'
    "TH 1e20\nNXT\nGLA 1.5\nRD 20\nTH 2\n"
    "NXT\nAIR\nRD -20\nTH 20\nNXT\nAIR\nEND 3\n"
)


def test_audit_counts_each_outcome_and_skips_non_lens_members(
    tmp_path, set_test_backend
):
    archive = tmp_path / "outcomes.zip"
    members = {
        "a-clean.LEN": CLEAN_LENS,
        "b-broken.len": 'LEN NEW "broken" 1 3\nRD invalid\n',
        "c-warned.len": CLEAN_LENS.replace("RD 20", "UNMAPPED\nRD 20"),
        "readme.txt": "Not an optical prescription",
    }
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)
    report = audit(archive, be.get_backend())
    clean, broken, warned = report["files"]
    assert [row["file"] for row in report["files"]] == sorted(members)[:3]
    assert report["summary"] == {"clean": 1, "failed": 1, "warned": 1}
    assert report["strict_successes"] == 1
    assert report["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert clean["sha256"] == hashlib.sha256(CLEAN_LENS.encode()).hexdigest()
    assert clean["warnings"] == []
    assert (
        clean["strict_error"] is clean["import_error"] is clean["trace_error"] is None
    )
    assert clean["finite_transmitted_rays"] == 5
    assert broken["import_error"].startswith("ValueError: b-broken.len:")
    assert broken["strict_error"].startswith("ValueError: b-broken.len:")
    assert "trace_error" not in broken
    assert "finite_transmitted_rays" not in broken
    assert "UNMAPPED" in warned["strict_error"]
    assert warned["import_error"] is warned["trace_error"] is None
    assert "oslo-audit-" not in json.dumps(report)


def test_audit_records_trace_failure_without_changing_import_status(
    tmp_path, monkeypatch, set_test_backend
):
    archive = tmp_path / "trace.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("clean.len", CLEAN_LENS)

    def fail_trace(self, *args, **kwargs):
        raise RuntimeError("trace failed after import")

    monkeypatch.setattr(Optic, "trace", fail_trace)
    report = audit(archive, be.get_backend())
    row = report["files"][0]
    assert report["summary"] == {"clean": 1}
    assert report["strict_successes"] == 1
    assert row["import_error"] is row["strict_error"] is None
    assert row["trace_error"] == "RuntimeError: trace failed after import"
    assert "finite_transmitted_rays" not in row


def test_audit_cli_writes_full_report_and_prints_only_summary(
    tmp_path, monkeypatch, capsys
):
    archive = tmp_path / "cli.zip"
    output = tmp_path / "report.json"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("clean.len", CLEAN_LENS)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_oslo_examples", str(archive), str(output), "--backend", "numpy"],
    )
    # Execute the entry point in-process so CI coverage includes the CLI itself.
    runpy.run_path(audit_oslo_examples.__file__, run_name="__main__")
    report = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == {key: value for key, value in report.items() if key != "files"}
    assert report["backend"] == "numpy"
    assert report["summary"] == {"clean": 1}
    assert report["files"][0]["finite_transmitted_rays"] == 5


def test_archive_member_names_are_normalized_before_json_encoding(tmp_path):
    name = 'demos/quoted "lens"\\example.len'
    archive = tmp_path / "examples.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            name,
            'LEN NEW "audit" 50 3\nEBR 2\nANG 0\n'
            "TH 1e20\nNXT\nGLA 1.5\nRD 20\nTH 2\n"
            "UNMAPPED\nNXT\nAIR\nRD -20\nTH 20\nNXT\nAIR\nEND 3\n",
        )
        # ZipFile normalizes native directory separators when writing on Windows.
        name = zipped.namelist()[0]
    report = audit(archive, "numpy")
    row = json.loads(json.dumps(report))["files"][0]
    assert row["file"] == name
    assert row["import_error"] is None
    assert name in row["warnings"][0]
    assert name in row["strict_error"]
    assert "oslo-audit-" not in json.dumps(report)
    assert report["summary"] == {"warned": 1}
