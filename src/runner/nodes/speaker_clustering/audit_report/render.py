from __future__ import annotations

from html import escape
from pathlib import Path

from runner.nodes.speaker_clustering.audit_report.models import (
    AssignmentAuditBuildResult,
    AssignmentAuditDocument,
)


def render_assignment_audit(
    document: AssignmentAuditDocument,
    output_dir: Path,
) -> AssignmentAuditBuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit-report.json"
    html_path = output_dir / "audit-report.html"
    manifest_path = output_dir / "listening-manifest.json"
    report_json = document.model_dump_json(indent=2)
    manifest_json = document.listening_manifest.model_dump_json(indent=2)
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Speaker assignment audit</title></head><body>"
        "<h1>Speaker assignment audit</h1><pre>"
        f"{escape(report_json)}</pre></body></html>"
    )
    _atomic_write(json_path, report_json)
    _atomic_write(html_path, html)
    _atomic_write(manifest_path, manifest_json)
    return AssignmentAuditBuildResult(
        json_report_path=json_path,
        html_report_path=html_path,
        listening_manifest_path=manifest_path,
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
