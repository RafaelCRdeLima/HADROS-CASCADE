#!/usr/bin/env python3
"""Lightweight tests for final-pipeline required-output checks."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hadros_final_pipeline import outputs_ready


def test_outputs_ready_rejects_empty_required_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert not outputs_ready([path])


def test_outputs_ready_accepts_declared_empty_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "hits.jsonl"
        summary = Path(tmp) / "summary.csv"
        empty.write_text("", encoding="utf-8")
        summary.write_text("n\n0\n", encoding="utf-8")
        assert outputs_ready([empty, summary], allow_empty_outputs=(empty,))


def test_outputs_ready_rejects_missing_even_if_empty_allowed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.jsonl"
        assert not outputs_ready([missing], allow_empty_outputs=(missing,))


if __name__ == "__main__":
    test_outputs_ready_rejects_empty_required_output()
    test_outputs_ready_accepts_declared_empty_output()
    test_outputs_ready_rejects_missing_even_if_empty_allowed()
