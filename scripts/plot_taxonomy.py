"""Scientific plot taxonomy for HADROS-CASCADE dashboards and audits.

classify_plot() returns a dict compatible with build_run_plot_dashboard.py's
**taxonomy unpacking. dashboard_section_for() maps a category string to a
human-readable section heading.
"""

from __future__ import annotations

from pathlib import Path


_SECTION_MAP = {
    "CORE_SCIENCE": "Main science figures",
    "SCIENCE_VALIDATION": "Validation and provenance",
    "PIPELINE_DIAGNOSTIC": "Pipeline diagnostics",
    "DEBUG_ONLY": "Debug/legacy figures",
    "LEGACY": "Debug/legacy figures",
    "OBSOLETE_REMOVE_OR_HIDE": "Debug/legacy figures",
    "KEEP_UNCLASSIFIED_REQUIRES_USER_REVIEW": "Debug/legacy figures",
}

_CORE_KEYWORDS = [
    "particle_ray_association", "cascade_origin", "ray_associated",
    "gbw_iim_rgb", "gbw_iim_ratio",
]

_VALIDATION_KEYWORDS = [
    "ray_origin", "ray_to_particle", "ray_contribution", "gbw_iim_ray",
    "gbw_iim_tau", "gbw_iim_pint", "nearest_ray", "direction_misalignment",
    "tolerance_observed", "cc_vs_nc", "energy_budget", "particle_transport",
    "cartesian_vs_zamo", "weighted_energy", "weighted_channel",
    "gbw_vs_iim", "validation", "validate", "comparison", "convergence",
    "residual", "difference", "ratio", "attenuation", "psurv", "pint",
    "tau", "column", "spectrum", "budget", "survival", "fgar",
    "analytic", "benchmark",
]

_DIAGNOSTIC_KEYWORDS = [
    "interaction_points", "global_exit_positions", "column_model",
    "pint_model", "pdg_distribution", "multiplicity_distribution",
    "geant4_pass_fail", "theta_scan", "resolution_convergence",
    "torus_screen_area", "funnel_screen_area", "diagnostic", "distribution",
    "status", "positions", "scan", "profile", "surface", "density",
    "temperature", "ye", "opacity", "neutrinosphere", "torus", "funnel",
    "morphology", "geometry", "camera_rays", "query", "deposition",
]

_DEBUG_KEYWORDS = [
    "proxy", "hybrid", "autoframe", "packet", "directional_screen",
    "screen_projection", "local_response", "hybrid_packet", "packet_screen",
    "forward_packet", "straight_line", "legacy_packet", "cuda",
]

_LEGACY_KEYWORDS = [
    "backward_gamma", "backward_electromagnetic", "backward_hadronic",
    "backward_deposited", "particle_channel_images", "real_kerr_observed",
    "observed_rgb", "observed_gamma", "observed_hadronic", "observed_neutrino",
    "observed_electromagnetic", "observed_particles_count",
    "observed_particle_channel", "observed_particle_pdg",
]

_FINAL_CHAIN_KEYWORDS = [
    "real_kerr", "particle_ray_association", "observed_particles_by_pixel", "gbw_iim_real_kerr",
    "incoming_geodesic", "incoming_ray", "zamo",
    "powheg_pythia_geant4_resumable", "powheg_pythia_geant4", "powheg_geant4",
]


def classify_plot(path_or_name: str | Path, context: str = "") -> dict[str, object]:
    path = Path(path_or_name)
    raw_name = path.name
    name = raw_name.lstrip(">` '\"")
    lower = name.lower()
    path_lower = path.as_posix().lower()
    context_lower = f"{path_lower} {context.lower()}"

    uses_proxy = False

    if any(token in context_lower for token in _DEBUG_KEYWORDS):
        category = "DEBUG_ONLY"
        importance = "D"
        visibility = "DEBUG"
        uses_proxy = True
        action = "Show only in Debug with DEBUG ONLY badge."

    elif raw_name != name or "{" in lower or "}" in lower or "<" in lower:
        category = "KEEP_UNCLASSIFIED_REQUIRES_USER_REVIEW"
        importance = "E"
        visibility = "HIDE"
        action = "Malformed or template plot; keep hidden until reviewed."

    elif any(token in lower for token in _CORE_KEYWORDS):
        category = "CORE_SCIENCE"
        importance = "A"
        visibility = "NORMAL"
        action = "Show in main Outputs and dashboard."

    elif any(token in lower for token in _LEGACY_KEYWORDS):
        category = "LEGACY"
        importance = "D"
        visibility = "DEBUG"
        uses_proxy = "proxy" in lower
        action = "Keep hidden; legacy plot not part of the final chain."

    elif any(token in lower for token in _FINAL_CHAIN_KEYWORDS):
        if any(t in lower for t in ["observed_", "rgb", "channel_histogram", "pdg_histogram"]) and "test_" not in lower:
            category = "CORE_SCIENCE"
            importance = "A"
            visibility = "NORMAL"
            action = "Show in normal Outputs when produced by the final chain."
        elif any(t in lower for t in _VALIDATION_KEYWORDS):
            category = "SCIENCE_VALIDATION"
            importance = "B"
            visibility = "EXPERT"
            action = "Show in Expert validation sections."
        else:
            category = "PIPELINE_DIAGNOSTIC"
            importance = "C"
            visibility = "EXPERT"
            action = "Expert diagnostic for the final ray-linked chain."

    elif any(token in lower for token in _VALIDATION_KEYWORDS):
        category = "SCIENCE_VALIDATION"
        importance = "B"
        visibility = "EXPERT"
        action = "Show in Expert as validation; not a main science figure."

    elif any(token in lower for token in _DIAGNOSTIC_KEYWORDS):
        category = "PIPELINE_DIAGNOSTIC"
        importance = "C"
        visibility = "EXPERT"
        action = "Show in Expert diagnostics."

    elif lower.endswith(".pdf") or "logo" in lower or "manual" in lower:
        category = "OBSOLETE_REMOVE_OR_HIDE"
        importance = "E"
        visibility = "HIDE"
        action = "Documentation/static asset; hide from plot UI."

    else:
        category = "KEEP_UNCLASSIFIED_REQUIRES_USER_REVIEW"
        importance = "E"
        visibility = "HIDE"
        action = "Keep hidden and require user review before promotion."

    return {
        "category": category,
        "importance": importance,
        "visibility": visibility,
        "uses_final_chain": category in {"CORE_SCIENCE", "SCIENCE_VALIDATION"} and any(
            t in context_lower for t in _FINAL_CHAIN_KEYWORDS + ["gbw_iim", "observed_"]
        ),
        "uses_proxy_or_legacy": uses_proxy or category in {"DEBUG_ONLY", "LEGACY", "OBSOLETE_REMOVE_OR_HIDE"},
        "scientific_claim_allowed": category in {"CORE_SCIENCE", "SCIENCE_VALIDATION"},
        "recommended_action": action,
        "caption": _caption_for(name, category),
    }


def _caption_for(name: str, category: str) -> str:
    if category == "CORE_SCIENCE":
        return f"{name} shows a main HADROS-CASCADE scientific output from the real Kerr ray-linked particle/GBW-IIM chain."
    if category == "SCIENCE_VALIDATION":
        return f"{name} validates provenance, transport, or reweighting consistency for the final cascade chain."
    if category == "PIPELINE_DIAGNOSTIC":
        return f"{name} is a technical diagnostic for troubleshooting cascade pipeline stages."
    if category in {"DEBUG_ONLY", "LEGACY"}:
        return f"{name} is DEBUG ONLY and is not part of the final scientific cascade chain."
    if category == "OBSOLETE_REMOVE_OR_HIDE":
        return f"{name} is obsolete or hidden pending removal/review."
    return f"{name} remains hidden until user review confirms its scientific role."


def dashboard_section_for(category: str, name: str = "") -> str:
    lower = name.lower()
    if category == "CORE_SCIENCE":
        if "gbw" in lower or "iim" in lower or "ratio" in lower:
            return "GBW/IIM comparison"
        if "histogram" in lower or "pdg" in lower or "channel" in lower:
            return "Particle composition"
        return "Main science figures"
    if category == "SCIENCE_VALIDATION":
        return "Validation and provenance"
    if category == "PIPELINE_DIAGNOSTIC":
        return (
            "GEANT4 transport diagnostics"
            if "geant4" in lower or "energy_budget" in lower
            else "Pipeline diagnostics"
        )
    return "Debug/legacy figures"
