#!/usr/bin/env python3
"""Build a run-local dashboard using only plots inside output/<RUN_NAME>/plots."""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path

from hadros_captions import caption_for_plot
from plot_taxonomy import classify_plot, dashboard_section_for


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
LEGACY_HIDDEN = {
    "tau_2d_map_rho0_vs_energy.png",
}


def sanitize_run_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()).strip("._-")
    return cleaned or "run_001"


def display_name_for_run_dir(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(ROOT / "output").as_posix()
    except ValueError:
        return run_dir.name


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def title_from_path(path: Path) -> str:
    replacements = {
        "uhe": "UHE",
        "mev": "MeV",
        "gbw": "GBW",
        "iim": "IIM",
        "ctw": "CTW",
        "tau": "tau",
        "psurv": "P_surv",
        "rgb": "RGB",
    }
    words = []
    for word in path.stem.replace("-", "_").split("_"):
        if not word:
            continue
        lower = word.lower()
        words.append(replacements.get(lower, word.capitalize()))
    return " ".join(words) or path.name


def description_for(path: Path) -> str:
    name = path.name.lower()
    if "fig_tau_phase_gbw_stream" in name:
        return "Physical stream-mode UHE opacity phase map for the GBW DIS model."
    if "fig_tau_phase_iim_stream" in name:
        return "Physical stream-mode UHE opacity phase map for the IIM DIS model."
    if "fig_tau_phase_iim_over_gbw" in name:
        return "Ratio of physical stream-mode UHE opacity metrics, IIM divided by GBW."
    if "tau_3model" in name:
        return "DIS model comparison for UHE optical depth."
    if "tau_2d" in name:
        return "UHE optical-depth phase diagram."
    if "attenuated_spectrum" in name:
        return "UHE spectral attenuation plot."
    if "psurv_vs_inclination" in name:
        return "Survival probability versus observer inclination."
    if "photon_ring_profile" in name:
        return "Photon-ring or image-profile diagnostic for this run."
    if "combined" in name or "rgb" in name:
        return "UHE/MeV image-comparison product."
    if "contour" in name:
        return "Image contour diagnostic."
    if "uhe" in name:
        return "UHE image or diagnostic plot."
    if "mev" in name:
        return "MeV image or diagnostic plot."
    return "Run-local scientific plot."


def channel_section_for(item: dict[str, str]) -> str:
    category = str(item.get("category", ""))
    if category:
        return dashboard_section_for(category, item.get("filename", ""))
    name = Path(item.get("filename", "")).name.lower()
    module = str(item.get("module", "")).lower()
    plot_type = str(item.get("plot_type", "")).lower()
    if "momentum" in name or "direction" in name or "angular" in name or "anisotropy" in name or "momentum" in plot_type:
        return "Momentum/direction diagnostics"
    if "hadros_backward" in name or "channel_" in name or "particle" in name or "cascade" in module:
        return "Camera-observed cascade particle-channel maps"
    if "mev" in name:
        return "Camera-observed MeV neutrino maps"
    if "uhe" in name or "tau" in name or "psurv" in name or "attenuated" in name:
        return "Camera-observed UHE neutrino maps"
    return "Deprecated/legacy diagnostics"


def rel_from(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def manifest_plots(run_dir: Path, dashboard_dir: Path) -> list[dict[str, str]]:
    """
    @brief Read plot entries from a run manifest for dashboard rendering.
    @param run_dir Root directory of the HADROS run.
    @param dashboard_dir Directory where the dashboard HTML is written.
    @return Plot entries enriched with links and LaTeX captions.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    entries: list[dict[str, str]] = []
    for item in payload.get("files", []):
        if item.get("kind") != "plot":
            continue
        rel_path = str(item.get("path", ""))
        if not rel_path:
            continue
        path = run_dir / rel_path
        if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS or path.name in LEGACY_HIDDEN:
            continue
        taxonomy = classify_plot(rel_path)
        entry = {
            "title": title_from_path(path),
            "filename": rel_path,
            "path": path.relative_to(ROOT).as_posix(),
            "href": rel_from(path, dashboard_dir),
            "description": description_for(path),
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "module": str(item.get("module", "")),
            "plot_type": str(item.get("plot_type", "")),
            "caption_latex": str(item.get("caption_latex", "")),
            "label_latex": str(item.get("label_latex", "")),
            "caption_source": str(item.get("caption_source", "")),
            "relevant_parameters": item.get("relevant_parameters", {}),
            **taxonomy,
        }
        if not entry["caption_latex"] or not entry["label_latex"]:
            entry.update(
                caption_for_plot(
                    path=rel_path,
                    module=entry["module"],
                    role=str(item.get("role", "")),
                    config=payload.get("config", {}).get("values", {}),
                )
            )
        entries.append(entry)
    return sorted(entries, key=lambda item: item["filename"])


def write_plot_manifest(run_dir: Path, entries: list[dict[str, str]]) -> None:
    fields = [
        "plot_path",
        "plot_type",
        "category",
        "importance",
        "visibility",
        "uses_final_chain",
        "uses_proxy_or_legacy",
        "scientific_claim_allowed",
        "caption",
    ]
    rows = []
    for entry in entries:
        taxonomy = classify_plot(entry.get("filename", ""))
        rows.append(
            {
                "plot_path": entry.get("filename", ""),
                "plot_type": entry.get("plot_type", Path(entry.get("filename", "")).suffix.lower().lstrip(".")),
                "category": entry.get("category", taxonomy["category"]),
                "importance": entry.get("importance", taxonomy["importance"]),
                "visibility": entry.get("visibility", taxonomy["visibility"]),
                "uses_final_chain": str(entry.get("uses_final_chain", taxonomy["uses_final_chain"])),
                "uses_proxy_or_legacy": str(entry.get("uses_proxy_or_legacy", taxonomy["uses_proxy_or_legacy"])),
                "scientific_claim_allowed": str(entry.get("scientific_claim_allowed", taxonomy["scientific_claim_allowed"])),
                "caption": entry.get("caption", taxonomy["caption"]),
            }
        )
    with (run_dir / "plot_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# HADROS Plot Manifest", ""]
    if not rows:
        lines.append("No plot products were indexed for this run.")
    else:
        lines.extend([
            "| plot_path | category | importance | visibility | scientific_claim_allowed |",
            "|---|---|---|---|---|",
        ])
        for row in rows:
            lines.append(
                f"| `{row['plot_path']}` | {row['category']} | {row['importance']} | {row['visibility']} | {row['scientific_claim_allowed']} |"
            )
    (run_dir / "plot_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def visible_for_dashboard(entry: dict[str, str], visibility_mode: str) -> bool:
    visibility = str(entry.get("visibility", "EXPERT"))
    mode = visibility_mode.upper()
    if mode == "DEBUG":
        return True
    if visibility in {"DEBUG", "HIDE"}:
        return False
    if mode == "EXPERT":
        return visibility in {"NORMAL", "EXPERT"}
    return visibility == "NORMAL"


def read_csv_preview(path: Path, limit: int = 8) -> tuple[list[str], list[dict[str, str]]]:
    import csv

    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
        return list(reader.fieldnames or []), rows


def render_table_preview(title: str, fields: list[str], rows: list[dict[str, str]]) -> str:
    if not fields:
        return ""
    useful_fields = fields[:6]
    header = "".join(f"<th>{html.escape(field)}</th>" for field in useful_fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in useful_fields) + "</tr>"
        for row in rows
    )
    return f"<h3>{html.escape(title)}</h3><div class=\"table-scroll\"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def render_observed_particles_section(run_dir: Path, dashboard_dir: Path, run_manifest: dict[str, object] | None = None) -> str:
    run_manifest = run_manifest or {}
    particle_requested = bool(run_manifest.get("produce_uhe_collision_particles") or run_manifest.get("particle_production") or run_manifest.get("run_uhe_particle_cascade"))
    cascade_dir = run_dir / "cascade"
    product_dir = cascade_dir if ((cascade_dir / "particle_ray_association_camera.csv").exists() or (cascade_dir / "observed_particles_by_pixel.csv").exists()) else run_dir
    particles = product_dir / "particle_ray_association_camera.csv"
    legacy_particles = product_dir / "observed_particles_by_pixel.csv"
    if not particles.exists():
        particles = legacy_particles
    if not particles.exists():
        if particle_requested:
            return """
            <section class="plot-section particle-ray-association error-section">
              <h2>Particle-Ray Association Maps</h2>
              <p><strong>Particle production was requested, but no particle-ray association products were generated.</strong></p>
            </section>
            """
        return """
        <section class="plot-section particle-ray-association">
          <h2>Particle-Ray Association Maps</h2>
          <p>This run contains UHE DIS neutrino products only. No secondary particle production was requested.</p>
        </section>
        """
    pdg_hist = product_dir / "particle_ray_association_pdg_histogram.csv"
    channel_hist = product_dir / "particle_ray_association_channel_histogram.csv"
    if not pdg_hist.exists():
        pdg_hist = product_dir / "observed_particle_pdg_histogram.csv"
    if not channel_hist.exists():
        channel_hist = product_dir / "observed_particle_channel_histogram.csv"
    fields_pdg, rows_pdg = read_csv_preview(pdg_hist)
    fields_ch, rows_ch = read_csv_preview(channel_hist)
    links = []
    for rel in [
        "particle_ray_association_camera.csv",
        "particle_ray_association_camera.jsonl",
        "particle_ray_association_camera_summary.md",
        "particle_ray_association_pdg_histogram.csv",
        "particle_ray_association_channel_histogram.csv",
        "observed_particles_by_pixel.csv",
        "observed_particles_by_pixel.jsonl",
        "observed_particles_by_pixel_summary.md",
        "observed_particle_pdg_histogram.csv",
        "observed_particle_channel_histogram.csv",
    ]:
        path = product_dir / rel
        if path.exists():
            links.append(f'<a href="{html.escape(rel_from(path, dashboard_dir))}">{html.escape(rel)}</a>')
    plot_links = []
    for pattern in ["particle_ray_association*.png", "observed_*.png"]:
        for path in sorted((product_dir / "plots").glob(pattern)):
            plot_links.append(f'<a href="{html.escape(rel_from(path, dashboard_dir))}">{html.escape(path.name)}</a>')
    return f"""
    <section class="plot-section particle-ray-association">
      <h2>Particle-Ray Association Maps</h2>
      <p>Pixel-level cascade-origin association product. Legacy <code>observed_particles_by_pixel.*</code> names are compatibility outputs only and do not imply full secondary-particle transport to the observer.</p>
      <div class="link-row">{' · '.join(links)}</div>
      <div class="link-row">{' · '.join(plot_links)}</div>
      {render_table_preview('Ray-associated PDGs', fields_pdg, rows_pdg)}
      {render_table_preview('Ray-associated channels', fields_ch, rows_ch)}
    </section>
    """


def render_scientific_bundle_section(run_dir: Path, dashboard_dir: Path, visibility_mode: str = "NORMAL") -> str:
    manifest_path = run_dir / "plots" / "science" / "plot_bundle_manifest.csv"
    if not manifest_path.exists():
        return ""
    fields, rows = read_csv_preview(manifest_path, limit=200)
    if not rows:
        return ""
    ordered = sorted(rows, key=lambda row: row.get("plot_id", row.get("filename", "")))
    cards = []
    for row in ordered:
        filename = row.get("filename", "")
        if not filename:
            continue
        path = run_dir / "plots" / "science" / filename
        href = rel_from(path, dashboard_dir)
        exists = row.get("exists", "").lower() == "true" or row.get("generated") == "True"
        if exists and path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            preview = f'<a href="{html.escape(href)}"><img src="{html.escape(href)}" alt="{html.escape(filename)}"></a>'
        else:
            preview = f'<div class="file-preview">Missing: {html.escape(row.get("missing_reason", ""))}</div>'
        cards.append(
            f"""
            <article class="card">
              <div class="preview">{preview}</div>
              <div class="body">
                <h3>{html.escape(filename)}</h3>
                <p>{html.escape(row.get('description', ''))}</p>
                <dl>
                  <dt>Source</dt><dd>{html.escape(row.get('source_data', ''))}</dd>
                  <dt>Exists</dt><dd>{html.escape(row.get('exists', row.get('generated', '')))}</dd>
                </dl>
              </div>
            </article>
            """
        )
    if not cards:
        return ""
    return f"""
    <section class="plot-section scientific-bundle">
      <h2>Paper-ready science figures</h2>
      <div class="grid">{''.join(cards)}</div>
    </section>
    """


def render_dashboard(
    run_name: str,
    plots: list[Path],
    manifest: list[dict[str, str]],
    run_manifest: dict[str, object] | None = None,
    run_dir: Path | None = None,
    dashboard_dir: Path | None = None,
    visibility_mode: str = "NORMAL",
) -> str:
    """
    @brief Render the static run-local HTML dashboard.
    @param run_name Sanitized run name.
    @param plots Plot paths that will be displayed.
    @param manifest Plot metadata entries from `manifest.json`.
    @return Complete offline HTML document as a string.
    """
    section_names = [
        "Main science figures",
        "GBW/IIM comparison",
        "Particle composition",
        "Validation and provenance",
        "GEANT4 transport diagnostics",
        "Debug/legacy figures",
    ]
    cards_by_section: dict[str, list[str]] = {name: [] for name in section_names}
    visible_manifest = [item for item in manifest if visible_for_dashboard(item, visibility_mode)]
    for index, item in enumerate(visible_manifest):
        plot_name = item["filename"]
        href = item.get("href", f"../plots/{plot_name}")
        caption = item.get("caption_latex", "")
        label = item.get("label_latex", "")
        category = str(item.get("category", "UNKNOWN_REVIEW"))
        visibility = str(item.get("visibility", "EXPERT"))
        badge = ""
        if category in {"DEBUG_ONLY", "LEGACY", "OBSOLETE_REMOVE_OR_HIDE"} or visibility in {"DEBUG", "HIDE"}:
            badge = '<div class="debug-badge">DEBUG ONLY — not part of final scientific chain</div>'
        figure_block = (
            "\\begin{figure}[htbp]\n"
            "    \\centering\n"
            f"    \\includegraphics[width=\\linewidth]{{{plot_name}}}\n"
            f"    {caption}\n"
            f"    {label}\n"
            "\\end{figure}"
        )
        copy_id = f"caption-{index}"
        suffix = Path(plot_name).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            preview = f'<a href="{html.escape(href)}"><img src="{html.escape(href)}" alt="{html.escape(item["title"])}"></a>'
        else:
            preview = f'<a class="file-preview" href="{html.escape(href)}">Open figure</a>'
        card = (
            f"""
            <article class="card">
              <div class="preview">{preview}</div>
              <div class="body">
                {badge}
                <h3>{html.escape(item["title"])}</h3>
                <p>{html.escape(item["description"])}</p>
                <dl>
                  <dt>File</dt><dd>{html.escape(plot_name)}</dd>
                  <dt>Category</dt><dd>{html.escape(category)}</dd>
                  <dt>Visibility</dt><dd>{html.escape(visibility)}</dd>
                  <dt>Module</dt><dd>{html.escape(item.get("module", ""))}</dd>
                  <dt>Type</dt><dd>{html.escape(item.get("plot_type", ""))}</dd>
                  <dt>Updated</dt><dd>{html.escape(item["updated_at"])}</dd>
                </dl>
                <label class="caption-label" for="{copy_id}">LaTeX figure block</label>
                <pre class="caption-box" id="{copy_id}">{html.escape(figure_block)}</pre>
                <button class="copy-caption" data-copy-target="{copy_id}">Copy LaTeX caption</button>
              </div>
            </article>
            """
        )
        cards_by_section.setdefault(channel_section_for(item), []).append(card)
    section_blocks = []
    for section_name in section_names:
        cards = cards_by_section.get(section_name, [])
        if not cards:
            continue
        section_blocks.append(
            f"""
            <section class="plot-section">
              <h2>{html.escape(section_name)}</h2>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )
    empty = ""
    if not section_blocks:
        empty = f"<p>No plots found in <code>output/{html.escape(run_name)}/plots/</code>.</p>"
    observed_particles_section = render_observed_particles_section(run_dir, dashboard_dir, run_manifest) if run_dir and dashboard_dir else ""
    scientific_bundle_section = render_scientific_bundle_section(run_dir, dashboard_dir, visibility_mode) if run_dir and dashboard_dir else ""
    run_manifest = run_manifest or {}
    observed_particles = run_manifest.get("observed_particles") or []
    if isinstance(observed_particles, list):
        observed_text = ", ".join(str(item) for item in observed_particles) or "unknown"
    else:
        observed_text = str(observed_particles)
    channel_summary = f"""
    <div class="channel-summary">
      <strong>What does the particle-ray association camera map?</strong>
      <span>particle/channel: {html.escape(observed_text)}</span>
      <span>energy: {html.escape(str(run_manifest.get('observed_energy_min', '')))} - {html.escape(str(run_manifest.get('observed_energy_max', '')))}</span>
      <span>momentum: {html.escape(str(run_manifest.get('observed_momentum_mode', 'integrated')))}</span>
      <span>required modules: {html.escape(', '.join(str(item) for item in run_manifest.get('required_modules', [])) or 'unknown')}</span>
      <span>UHE DIS: {'yes' if run_manifest.get('run_uhe_dis') else 'no'}</span>
      <span>produce UHE collision particles: {'yes' if run_manifest.get('produce_uhe_collision_particles') or run_manifest.get('particle_production') or run_manifest.get('run_uhe_particle_cascade') else 'no'}</span>
      <span>MeV torus neutrinos: {'yes' if run_manifest.get('run_mev_torus_neutrinos') else 'no'}</span>
    </div>
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HADROS Run Plots - {html.escape(run_name)}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #172033; }}
    header {{ padding: 24px clamp(18px, 4vw, 46px); background: #fff; border-bottom: 1px solid #d9dee7; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(24px, 4vw, 38px); }}
    header p {{ margin: 0; color: #657182; }}
    main {{ padding: 24px clamp(18px, 4vw, 46px); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }}
    .plot-section {{ margin: 0 0 26px; }}
    .plot-section h2 {{ margin: 0 0 12px; font-size: 21px; }}
    .channel-summary {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 12px; color: #344054; }}
    .channel-summary span {{ background: #eef2f6; border: 1px solid #d9dee7; border-radius: 6px; padding: 3px 7px; }}
    .card {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }}
    .preview {{ background: #eef2f6; min-height: 180px; display: grid; place-items: center; }}
    .preview img {{ width: 100%; height: 260px; object-fit: contain; display: block; }}
    .body {{ padding: 14px 16px 16px; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; }}
    p {{ margin: 0 0 12px; line-height: 1.45; }}
    dl {{ display: grid; grid-template-columns: 72px 1fr; gap: 5px 10px; margin: 0; font-size: 13px; }}
    dt {{ color: #657182; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
    .file-preview {{ padding: 16px; }}
    .caption-label {{ display: block; margin-top: 12px; font-size: 12px; color: #657182; font-weight: 650; }}
    .caption-box {{ margin: 6px 0 10px; padding: 10px; max-height: 190px; overflow: auto; white-space: pre-wrap; border: 1px solid #d9dee7; border-radius: 6px; background: #f8fafc; font-size: 12px; line-height: 1.35; }}
    .copy-caption {{ border: 1px solid #b9c2d0; border-radius: 6px; background: #fff; color: #172033; padding: 7px 10px; cursor: pointer; font: inherit; font-size: 13px; }}
    .copy-caption:focus {{ outline: 2px solid #7895ff; outline-offset: 2px; }}
    .debug-badge {{ display: inline-block; margin: 0 0 8px; padding: 4px 7px; border-radius: 5px; background: #fff4e5; border: 1px solid #f0b35b; color: #7a4100; font-size: 12px; font-weight: 750; }}
    .link-row {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 10px 0 14px; }}
    .table-scroll {{ overflow-x: auto; margin: 8px 0 16px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dee7; }}
    th, td {{ padding: 7px 9px; border-bottom: 1px solid #e6ebf2; text-align: left; font-size: 13px; }}
    th {{ background: #eef2f6; color: #344054; }}
  </style>
</head>
<body>
  <header>
    <h1>HADROS Run Plots</h1>
    <p>Run: <strong>{html.escape(run_name)}</strong> · source folder: <code>output/{html.escape(run_name)}/plots/</code> · dashboard mode: <strong>{html.escape(visibility_mode.upper())}</strong> · plots indexed: {len(visible_manifest)} / {len(manifest)}</p>
    {channel_summary}
  </header>
  <main>
    {empty}
    {scientific_bundle_section}
    {observed_particles_section}
    {''.join(section_blocks)}
  </main>
  <script>
    function fallbackCopy(text) {{
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.select();
      try {{ document.execCommand("copy"); }} finally {{ document.body.removeChild(area); }}
    }}
    document.querySelectorAll(".copy-caption").forEach((button) => {{
      button.addEventListener("click", async () => {{
        const target = document.getElementById(button.dataset.copyTarget);
        const text = target ? target.textContent : "";
        try {{
          if (navigator.clipboard && window.isSecureContext) {{
            await navigator.clipboard.writeText(text);
          }} else {{
            fallbackCopy(text);
          }}
          button.textContent = "Copied";
          setTimeout(() => button.textContent = "Copy LaTeX caption", 1400);
        }} catch (error) {{
          if (target) {{
            const range = document.createRange();
            range.selectNodeContents(target);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
          }}
          button.textContent = "Select and copy";
        }}
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    """
    @brief Command-line entry point for run-local dashboard generation.
    @return Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="run_001")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional explicit output run directory.")
    parser.add_argument("--visibility", choices=["NORMAL", "EXPERT", "DEBUG"], default="NORMAL", help="Dashboard plot visibility level.")
    args = parser.parse_args()
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        run_name = display_name_for_run_dir(run_dir)
    else:
        run_name = sanitize_run_name(args.run_name)
        run_dir = ROOT / "output" / run_name
    dashboard_dir = run_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = run_dir / "plots"
    run_manifest = {}
    for manifest_name in ("run_manifest.json", "cascade/run_manifest.json"):
        candidate = run_dir / manifest_name
        if candidate.exists():
            try:
                run_manifest = json.loads(candidate.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                run_manifest = {}
    manifest = manifest_plots(run_dir, dashboard_dir)
    plots = [run_dir / item["filename"] for item in manifest]
    cascade_plot_dir = run_dir / "cascade" / "plots"
    if not manifest and (plot_dir.exists() or cascade_plot_dir.exists()):
        plots = sorted(
            path
            for base in (plot_dir, cascade_plot_dir)
            if base.exists()
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.name not in LEGACY_HIDDEN
        )
        manifest = [
            {
                "title": title_from_path(path),
                "filename": path.relative_to(run_dir).as_posix(),
                "path": display_path(path),
                "href": rel_from(path, dashboard_dir),
                "description": description_for(path),
                "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "module": "legacy",
                **classify_plot(path.relative_to(run_dir).as_posix()),
                **caption_for_plot(
                    path=path.relative_to(run_dir).as_posix(),
                    module="legacy",
                    role=path.stem,
                ),
            }
            for path in plots
        ]
    write_plot_manifest(run_dir, manifest)
    (dashboard_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dashboard_dir / "index.html").write_text(
        render_dashboard(run_name, plots, manifest, run_manifest, run_dir, dashboard_dir, args.visibility),
        encoding="utf-8",
    )
    print(f"Run plot dashboard written: {display_path(dashboard_dir / 'index.html')}")
    print(f"Plots indexed from run folder: {len(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
