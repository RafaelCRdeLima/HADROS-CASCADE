"""Run-local output helpers for HADROS scripts.

This module keeps output layout decisions in one place for Python workflows
driven by config-web.  It creates subdirectories only when a caller requests a
file in that sub-tree, and records generated products in output/<RUN>/manifest.json.
"""

from __future__ import annotations

import configparser
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from hadros_captions import caption_for_plot
    from hadros_metadata import build_plot_metadata, relevant_parameters, write_metadata
except ImportError:  # pragma: no cover - supports package-style imports in tests.
    from scripts.hadros_captions import caption_for_plot
    from scripts.hadros_metadata import build_plot_metadata, relevant_parameters, write_metadata


ROOT = Path(__file__).resolve().parents[1]


def sanitize_run_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()).strip("._-")
    return cleaned or "run_001"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _config_to_dict(path: Path) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)
    return {section: dict(parser.items(section)) for section in parser.sections()}


class HadrosOutputManager:
    """
    @brief Central manager for a single HADROS run output tree.

    The manager owns `output/<run_name>/`, creates subdirectories only on
    demand, snapshots configuration files, and records generated products in
    `manifest.json`.
    """

    def __init__(self, run_name: str, root: Path = ROOT) -> None:
        self.root = root
        self.run_name = sanitize_run_name(run_name)
        self.run_dir = self.root / "output" / self.run_name
        self.manifest_path = self.run_dir / "manifest.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """
        @brief Load an existing run manifest or initialize a new one.
        @return Mutable manifest dictionary for the current run.
        """
        if self.manifest_path.exists():
            try:
                payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload.setdefault("run_name", self.run_name)
                    payload.setdefault("created_at", _now_iso())
                    payload.setdefault("modules", [])
                    payload.setdefault("files", [])
                    payload.setdefault("messages", [])
                    payload.setdefault("status", "running")
                    return payload
            except json.JSONDecodeError:
                pass
        return {
            "run_name": self.run_name,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "status": "running",
            "config": {},
            "modules": [],
            "files": [],
            "messages": [],
        }

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError:
            try:
                return path.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                return str(path)

    def path(self, section: str, module: str, filename: str) -> Path:
        directory = self.run_dir / section / module
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    def data_dir(self, module: str) -> Path:
        """
        @brief Return a run-local data directory for a scientific module.
        @param module Module name such as `inclination_scan` or `tau_phase`.
        @return Created directory path under `data/<module>`.
        """
        directory = self.run_dir / "data" / module
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def plot_dir(self, module: str) -> Path:
        """
        @brief Return a run-local plot directory for a scientific module.
        @param module Module name used in the manifest.
        @return Created directory path under `plots/<module>`.
        """
        directory = self.run_dir / "plots" / module
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def config_dir(self) -> Path:
        directory = self.run_dir / "config"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def metadata_dir(self, module: str) -> Path:
        directory = self.run_dir / "metadata" / module
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def snapshot_config(self, config_path: Path) -> None:
        """
        @brief Copy the active configuration into the run and store parsed values.
        @param config_path Path to the source INI configuration.
        """
        if not config_path.exists():
            self.add_message(f"Config snapshot skipped; file not found: {config_path}")
            return
        target_ini = self.config_dir() / "run_config.ini"
        target_json = self.config_dir() / "run_config.json"
        target_ini.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        target_json.write_text(
            json.dumps(_config_to_dict(config_path), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.manifest["config"] = {
            "source": self._relative(config_path),
            "run_config_ini": self._relative(target_ini),
            "run_config_json": self._relative(target_json),
            "values": _config_to_dict(config_path),
        }
        self.register_file(target_ini, kind="config", module="config", role="run_config_ini")
        self.register_file(target_json, kind="config", module="config", role="run_config_json")

    def add_module(self, module: str) -> None:
        modules = self.manifest.setdefault("modules", [])
        if module not in modules:
            modules.append(module)

    def add_message(self, message: str) -> None:
        self.manifest.setdefault("messages", []).append({"time": _now_iso(), "message": message})

    def register_file(
        self,
        path: Path,
        *,
        kind: str,
        module: str,
        role: str = "",
        plot_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        @brief Register one generated run product in `manifest.json`.
        @param path File path to register.
        @param kind Product kind, e.g. `data`, `plot`, `metadata`, or `config`.
        @param module Scientific or workflow module that created the product.
        @param role Module-specific role for the file.
        @param plot_metadata Optional plot context used for automatic captions.
        """
        self.add_module(module)
        rel_path = self._relative(path)
        files = self.manifest.setdefault("files", [])
        files[:] = [
            item for item in files
            if not (item.get("path") == rel_path and item.get("kind") == kind and item.get("module") == module)
        ]
        entry = {
            "path": rel_path,
            "kind": kind,
            "module": module,
            "role": role,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "updated_at": _now_iso(),
        }
        if kind == "plot":
            metadata_context = dict(plot_metadata or {})
            data_file = metadata_context.pop("data_file", None)
            plot_type = role or path.stem
            metadata_payload = build_plot_metadata(
                run_name=self.run_name,
                module=module,
                product_type="plot",
                plot_type=plot_type,
                data_file=data_file,
                plot_file=rel_path,
                config=self.manifest.get("config", {}).get("values", {}),
                context=metadata_context,
            )
            metadata_path = self.metadata_dir(module) / f"{path.stem}_metadata.json"
            write_metadata(metadata_path, metadata_payload)
            metadata_rel = self._relative(metadata_path)
            entry["metadata_file"] = metadata_rel
            entry["relevant_parameters"] = relevant_parameters(metadata_payload)
            entry.update(
                caption_for_plot(
                    path=rel_path,
                    module=module,
                    role=role,
                    config=self.manifest.get("config", {}).get("values", {}),
                    metadata=metadata_payload,
                )
            )
            entry["metadata_file"] = metadata_rel
            entry["relevant_parameters"] = {
                **entry.get("relevant_parameters", {}),
                **relevant_parameters(metadata_payload),
            }
        files.append(entry)

    def finalize(self, status: str = "success") -> None:
        """
        @brief Write the manifest to disk with a final status.
        @param status Run status string, normally `success` or `failed`.
        """
        self.manifest["status"] = status
        self.manifest["updated_at"] = _now_iso()
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
