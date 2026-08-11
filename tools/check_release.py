#!/usr/bin/env python3
"""Fail closed when a GitHub release contains data, secrets, or local paths."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024

FORBIDDEN_DIRS = {
    "data",
    "data_external",
    "raw",
    "processed",
    "derived",
    "database",
    "databases",
    "analysis",
    "figures",
    "tables",
    "manuscript",
    "supplement",
    "preflight",
    "preflight_v2",
    "models",
    "predictions",
    "outputs",
    "results",
    ".venv",
    "venv",
    "env",
    "__pycache__",
}

FORBIDDEN_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arrow",
    ".xlsx",
    ".xls",
    ".xpt",
    ".sas7bdat",
    ".dta",
    ".rds",
    ".rdata",
    ".db",
    ".duckdb",
    ".sqlite",
    ".sqlite3",
    ".asc",
    ".gz",
    ".zip",
    ".tar",
    ".7z",
    ".joblib",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".dat",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".tif",
    ".tiff",
    ".docx",
    ".pptx",
}

FORBIDDEN_NAMES = {
    ".env",
    ".DS_Store",
    ".claude_resources.json",
    "pkcs11.txt",
}

CONTENT_PATTERNS = {
    "local macOS user path": re.compile("/" + "Users/"),
    "OpenAI-style secret": re.compile("sk" + r"-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
    "private key": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "assigned secret": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|passwd|secret)\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
}


def iter_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    )


def main() -> int:
    errors: list[str] = []
    files = iter_files()

    for path in files:
        rel = path.relative_to(ROOT)
        parts = set(rel.parts[:-1])
        forbidden_parts = parts & FORBIDDEN_DIRS
        if forbidden_parts:
            errors.append(f"forbidden directory {sorted(forbidden_parts)}: {rel}")

        if path.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden filename: {rel}")

        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file type {path.suffix}: {rel}")

        size = path.stat().st_size
        if size > MAX_BYTES:
            errors.append(f"oversized file ({size} bytes): {rel}")

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            errors.append(f"non-text or unreadable file: {rel}")
            continue

        for label, pattern in CONTENT_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {rel}")

    if errors:
        print("RELEASE BOUNDARY: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"RELEASE BOUNDARY: PASS ({len(files)} text files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
