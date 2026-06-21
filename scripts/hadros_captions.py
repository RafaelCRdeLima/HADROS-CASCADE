"""Central LaTeX caption registry for HADROS run-local plots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _label_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "hadros_plot"


def _config_value(config: dict[str, Any], *keys: str) -> str:
    for section in config.values():
        if not isinstance(section, dict):
            continue
        for key in keys:
            value = section.get(key)
            if value not in {None, ""}:
                return str(value)
    return ""


def _range_text(values: Any, unit: str = "") -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return ""
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return ""
    if len(nums) == 1 or min(nums) == max(nums):
        return f"{nums[0]:.3g}{unit}"
    return f"{min(nums):.3g}--{max(nums):.3g}{unit}"


def _grid_range_text(grid: dict[str, Any], unit_latex: str) -> str:
    lo = grid.get("min")
    hi = grid.get("max")
    if lo is None or hi is None:
        return ""
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except (TypeError, ValueError):
        return ""
    return rf"{_latex_number(lo_f)}--{_latex_number(hi_f)}\,{unit_latex}"


def _latex_number(value: float) -> str:
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1.0e4 or abs_value < 1.0e-2:
        exponent = int(__import__("math").floor(__import__("math").log10(abs_value)))
        mantissa = value / 10.0**exponent
        if abs(mantissa - 1.0) < 1.0e-10:
            return rf"10^{{{exponent}}}"
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    return f"{value:.4g}"


def _fmt(value: Any, precision: str = ".4g") -> str:
    if value is None:
        return "unknown"
    try:
        return format(float(value), precision)
    except (TypeError, ValueError):
        return str(value).replace("_", r"\_")


def _caption_payload(text: str, label: str, plot_type: str, relevant: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "plot_type": plot_type,
        "caption_latex": rf"\caption{{{text}}}",
        "label_latex": rf"\label{{{label}}}",
        "caption_source": source,
        "relevant_parameters": relevant,
    }


def _dynamic_tau_caption(path: Path, metadata: dict[str, Any], relevant: dict[str, Any]) -> dict[str, Any] | None:
    energy = metadata.get("energy_grid")
    density = metadata.get("density_grid")
    if not isinstance(energy, dict) or not isinstance(density, dict):
        return None
    cross = metadata.get("cross_section_parameters", {}) or {}
    bh = metadata.get("black_hole_parameters", {}) or {}
    torus = metadata.get("torus_parameters", {}) or {}
    image = (metadata.get("numerical_grid", {}) or {}).get("stream_image_grid") or {}
    model = cross.get("model") or ("IIM" if "iim" in path.name.lower() else "GBW" if "gbw" in path.name.lower() else "DIS")
    plot_type = "tau_phase_ratio" if "iim_over_gbw" in path.name.lower() else "tau_phase_diagram"
    quantity = metadata.get("metric") or metadata.get("physical_quantity") or "neutrino optical depth"
    energy_text = _grid_range_text(energy, r"\mathrm{GeV}")
    density_text = _grid_range_text(density, r"\mathrm{g\,cm^{-3}}")
    energy_points = energy.get("num_points") or "unknown"
    density_points = density.get("num_points") or "unknown"
    spacing_e = f"{energy.get('spacing')}arithmically spaced " if energy.get("spacing") == "log" else ""
    spacing_r = f"{density.get('spacing')}arithmically spaced " if density.get("spacing") == "log" else ""
    image_text = ""
    if isinstance(image, dict) and image.get("nx") and image.get("ny"):
        image_text = rf" Each phase point uses a stream image grid of {_fmt(image.get('nx'), '.0f')}\(\times\){_fmt(image.get('ny'), '.0f')} pixels."
    torus_rho = "unknown"
    if torus.get("rho0_gcm3") is not None:
        torus_rho = _latex_number(float(torus["rho0_gcm3"]))
    if plot_type == "tau_phase_ratio":
        text = (
            rf"Ratio of stream-mode optical-depth phase diagrams computed with the IIM and GBW dipole parametrizations "
            rf"of the charged-current DIS neutrino--nucleon cross section. The scan covers \(E_\nu={energy_text}\) "
            rf"with {energy_points} {spacing_e}energy points and \(\rho_0={density_text}\) with {density_points} "
            rf"{spacing_r}density points.{image_text} The color scale shows the model dependence of {quantity}."
        )
        label = rf"fig:tau_phase_iim_over_gbw_{_label_slug(str(metadata.get('run_name', 'run')))}"
    else:
        text = (
            rf"Optical-depth phase diagram for ultra-high-energy neutrinos computed with the {model} dipole parametrization "
            rf"of the charged-current DIS neutrino--nucleon cross section. The scan covers \(E_\nu={energy_text}\) "
            rf"with {energy_points} {spacing_e}energy points and \(\rho_0={density_text}\) with {density_points} "
            rf"{spacing_r}density points.{image_text} The color scale gives {quantity} for a Kerr black hole with "
            rf"\(M={_fmt(bh.get('mass_msun'))}\,M_\odot\) and spin \(a={_fmt(bh.get('spin'))}\), embedded in a torus "
            rf"with \(\rho_0={torus_rho}\,\mathrm{{g\,cm^{{-3}}}}\), "
            rf"\(r_0={_fmt(torus.get('r0_rg'))}\,r_g\), \(\sigma_r={_fmt(torus.get('sigma_r_rg'))}\,r_g\), "
            rf"and \(H/R={_fmt(torus.get('h_over_r'))}\). Contours mark transitions between optically thin and attenuated propagation regimes."
        )
        label = rf"fig:tau_phase_{_label_slug(str(model))}_{_label_slug(str(metadata.get('run_name', 'run')))}"
    return _caption_payload(text, label, plot_type, relevant, "metadata")


def _dynamic_inclination_caption(path: Path, metadata: dict[str, Any], relevant: dict[str, Any]) -> dict[str, Any] | None:
    scan = metadata.get("scan_parameters", {}) or {}
    grid = scan.get("inclination_grid") if isinstance(scan, dict) else None
    if not isinstance(grid, dict):
        return None
    bh = metadata.get("black_hole_parameters", {}) or {}
    torus = metadata.get("torus_parameters", {}) or {}
    camera = metadata.get("camera_parameters", {}) or {}
    theta_text = _grid_range_text(grid, r"^\circ")
    ntheta = grid.get("num_points") or "unknown"
    energies = scan.get("energies_GeV") or (metadata.get("physical_model", {}) or {}).get("energy_GeV")
    if isinstance(energies, list) and energies:
        if len(energies) == 1:
            energy_text = rf"{_latex_number(float(energies[0]))}\,\mathrm{{GeV}}"
        else:
            energy_text = _range_text(energies, r"\,\mathrm{GeV}")
    elif energies:
        energy_text = rf"{_latex_number(float(energies))}\,\mathrm{{GeV}}"
    else:
        energy_text = "the configured UHE energy"
    models = scan.get("models") or []
    model_text = ", ".join(str(item).replace("_", r"\_") for item in models) if models else "the configured DIS"
    image = (metadata.get("numerical_grid", {}) or {}).get("stream_image_grid") or {}
    image_text = ""
    if isinstance(image, dict) and image.get("nx") and image.get("ny"):
        image_text = rf" Stream images use {_fmt(image.get('nx'), '.0f')}\(\times\){_fmt(image.get('ny'), '.0f')} pixel sampling."
    torus_rho = "unknown"
    if torus.get("rho0_gcm3") is not None:
        torus_rho = _latex_number(float(torus["rho0_gcm3"]))
    survival_definition = str(
        scan.get("survival_probability_definition", "the stream-mode mean over valid rays")
    ).replace("_", r"\_")
    text = (
        rf"Survival probability of ultra-high-energy neutrinos as a function of observer inclination for the HADROS Kerr-torus configuration. "
        rf"The scan samples \(\theta_{{\rm obs}}={theta_text}\) using {ntheta} inclination angles and evaluates attenuation at \(E_\nu={energy_text}\). "
        rf"The propagation is computed with DIS neutrino--nucleon opacities using {model_text} parametrizations. "
        rf"The background model adopts a Kerr black hole with \(M={_fmt(bh.get('mass_msun'))}\,M_\odot\), spin \(a={_fmt(bh.get('spin'))}\), "
        rf"and a torus characterized by \(\rho_0={torus_rho}\,\mathrm{{g\,cm^{{-3}}}}\), "
        rf"\(r_0={_fmt(torus.get('r0_rg'))}\,r_g\), and \(H/R={_fmt(torus.get('h_over_r'))}\)."
        rf" The observer camera has \(r_{{\rm obs}}={_fmt(camera.get('r_obs_rg'))}\,r_g\) and field of view \({_fmt(camera.get('fov_deg'))}^\circ\)."
        rf"{image_text} Survival probability is defined as {survival_definition}."
    )
    label = rf"fig:inclination_scan_{_label_slug(str(metadata.get('run_name', 'run')))}"
    return _caption_payload(text, label, "inclination_survival_probability", relevant, "metadata")


def _dynamic_paper_caption(path: Path, metadata: dict[str, Any], relevant: dict[str, Any]) -> dict[str, Any] | None:
    module = str(metadata.get("module", "paper_result"))
    role = str(metadata.get("plot_type", path.stem))
    scan = metadata.get("scan_parameters", {}) or {}
    out = metadata.get("output_parameters", {}) or {}
    bh = metadata.get("black_hole_parameters", {}) or {}
    torus = metadata.get("torus_parameters", {}) or {}
    label = rf"fig:{_label_slug(path.stem + '_' + str(metadata.get('run_name', 'run')))}"
    name = path.name.lower()

    if "workflow" in name:
        text = (
            r"HADROS workflow used in the paper campaign. The pipeline starts from a run configuration and DIS cross-section tables, "
            r"constructs the Kerr-torus transport setup, computes optical-depth and survival-probability diagnostics, and records data, "
            r"plots, manifest entries, metadata JSON, dashboard cards, and LaTeX captions for reproducibility."
        )
        return _caption_payload(text, label, "workflow_schema", relevant, "metadata")

    if "cross_sections" in name and "ratios" not in name:
        models = ", ".join(str(x).replace("_", r"\_") for x in scan.get("models", [])) or "the configured models"
        rng = scan.get("energy_range_GeV", [])
        energy = ""
        if isinstance(rng, list) and len(rng) == 2:
            energy = rf" over \(E_\nu={_latex_number(float(rng[0]))}--{_latex_number(float(rng[1]))}\,\mathrm{{GeV}}\)"
        x_col = str(out.get('x_column', 'E')).replace('_', r'\_')
        y_col = str(out.get('y_column', 'sigma')).replace('_', r'\_')
        text = (
            rf"Charged-current UHE neutrino--nucleon DIS cross sections for {models}{energy}. "
            rf"The data are read from the HADROS sigma tables and plotted with log-log axes using "
            rf"\({x_col}\) and \({y_col}\)."
        )
        return _caption_payload(text, label, "dis_cross_section_comparison", relevant, "metadata")

    if "cross_section_ratios" in name:
        ratios = ", ".join(str(x).replace("_", r"\_") for x in scan.get("ratio_definitions", [])) or "model ratios"
        text = (
            rf"DIS cross-section ratios from the same tabulated inputs as the cross-section benchmark. "
            rf"The plotted ratios are {ratios}, isolating the hadronic-model dependence used later in the optical-depth calculations."
        )
        return _caption_payload(text, label, "dis_cross_section_ratio", relevant, "metadata")

    if "optical_depth_scaling" in name or "density_threshold" in name:
        energy = scan.get("fixed_energy_GeV")
        density = scan.get("density_range_gcm3", [])
        density_text = ""
        if isinstance(density, list) and len(density) == 2:
            density_text = rf" over \(\rho_0={_latex_number(float(density[0]))}--{_latex_number(float(density[1]))}\,\mathrm{{g\,cm^{{-3}}}}\)"
        text = (
            rf"Density-scaling optical-depth benchmark for the HADROS parametric Kerr-torus column model{density_text}. "
            rf"The calculation fixes \(E_\nu={_latex_number(float(energy))}\,\mathrm{{GeV}}\)" if energy else
            r"Density-scaling optical-depth benchmark for the HADROS parametric Kerr-torus column model"
        )
        text += (
            rf" and compares the GBW and IIM charged-current DIS tables. The dashed reference marks \(\tau_\nu=1\), "
            rf"which estimates the density threshold between transparent and attenuating propagation."
        )
        return _caption_payload(text, label, "density_scaling_validation", relevant, "metadata")

    if "survival_probability" in name:
        tau_range = scan.get("tau_range", [])
        range_text = ""
        if isinstance(tau_range, list) and len(tau_range) == 2:
            range_text = rf" for \(\tau_\nu={_latex_number(float(tau_range[0]))}--{_latex_number(float(tau_range[1]))}\)"
        text = (
            rf"Survival-probability validation{range_text}. The curve evaluates \(P_{{\rm surv}}=\exp(-\tau_\nu)\), "
            rf"recovering \(P_{{\rm surv}}\rightarrow1\) in the optically thin limit and \(P_{{\rm surv}}\rightarrow0\) in the optically thick limit."
        )
        return _caption_payload(text, label, "survival_probability_validation", relevant, "metadata")

    if "resolution_convergence" in name:
        ref = scan.get("reference_samples")
        text = (
            rf"Resolution-convergence benchmark for the inclination-averaged optical depth in the parametric Kerr-torus column model. "
            rf"The sampled mean \(\tau_\nu\) is compared against a reference calculation with {ref} inclination samples, and the ordinate gives the relative error."
        )
        return _caption_payload(text, label, "resolution_convergence", relevant, "metadata")

    if "stream_cache" in name:
        text = (
            r"Quantitative stream/cache-equivalent agreement test for the lightweight paper campaign. "
            r"The direct stream-style column integral is compared with a cached/interpolated column representation at matched inclinations; "
            r"the plotted quantity is the relative optical-depth error."
        )
        return _caption_payload(text, label, "stream_cache_quantitative_agreement", relevant, "metadata")

    if "geometry" in name or "h_over_r" in name:
        param = scan.get("parameter", "geometry parameter")
        rng = scan.get("range", [])
        rng_text = ""
        if isinstance(rng, list) and len(rng) == 2:
            rng_text = rf" over \(H/R={_fmt(rng[0])}--{_fmt(rng[1])}\)"
        param_label = str(param).replace('_', r'\_')
        text = (
            rf"Geometry-dependence scan varying {param_label}{rng_text}. "
            rf"The calculation keeps \(M={_fmt(bh.get('mass_msun'))}\,M_\odot\), spin \(a={_fmt(bh.get('spin'))}\), "
            rf"and \(\rho_0={_latex_number(float(torus.get('rho0_gcm3', 0.0)))}\,\mathrm{{g\,cm^{{-3}}}}\) fixed while comparing GBW and IIM opacities."
        )
        return _caption_payload(text, label, "geometry_dependence", relevant, "metadata")

    if "collapsar" in name or "application" in name:
        energy = scan.get("energy_range_GeV", [])
        incl = scan.get("inclination_range_deg", [])
        e_text = ""
        if isinstance(energy, list) and len(energy) == 2:
            e_text = rf" \(E_\nu={_latex_number(float(energy[0]))}--{_latex_number(float(energy[1]))}\,\mathrm{{GeV}}\)"
        i_text = ""
        if isinstance(incl, list) and len(incl) == 2:
            i_text = rf" and \(\theta_{{\rm obs}}={_fmt(incl[0])}--{_fmt(incl[1])}^\circ\)"
        text = (
            rf"Representative parametric collapsar/Kerr-torus application showing the survival probability over{e_text}{i_text}. "
            rf"The calculation uses the GBW charged-current DIS table, \(M={_fmt(bh.get('mass_msun'))}\,M_\odot\), spin \(a={_fmt(bh.get('spin'))}\), "
            rf"and a torus with \(\rho_0={_latex_number(float(torus.get('rho0_gcm3', 0.0)))}\,\mathrm{{g\,cm^{{-3}}}}\) and \(H/R={_fmt(torus.get('h_over_r'))}\). "
            rf"The color scale gives \(\log_{{10}}P_{{\rm surv}}\), separating transparent polar/low-energy regions from attenuated high-column regimes."
        )
        return _caption_payload(text, label, "collapsar_kerr_application", relevant, "metadata")

    return None


def caption_for_plot(
    *,
    path: str | Path,
    module: str,
    role: str = "",
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    @brief Generate LaTeX caption metadata for a HADROS plot.
    @param path Run-relative or filesystem path to the plot.
    @param module Module that generated the plot.
    @param role Optional plot role from the manifest.
    @param config Parsed run configuration values.
    @param metadata Optional module-specific plot metadata.
    @return Dictionary with plot type, caption, label, source, and parameters.
    """

    config = config or {}
    metadata = metadata or {}
    path_obj = Path(path)
    name = path_obj.name.lower()
    module = module or "unknown"
    stem = path_obj.stem
    plot_type = role or stem
    relevant: dict[str, str] = {}
    module_latex = module.replace("_", r"\_")

    spin = _config_value(config, "ASPIN")
    rho0 = _config_value(config, "TORUS_RHO0")
    density_profile = _config_value(config, "DENSITY_PROFILE")
    source_model = _config_value(config, "SOURCE_MODEL")
    sigma_path = _config_value(config, "SIGMA_TABLE_PATH")
    if spin:
        relevant["spin"] = spin
    if rho0:
        relevant["density_normalization"] = rho0
    if density_profile:
        relevant["density_profile"] = density_profile
    if source_model:
        relevant["source_model"] = source_model
    if sigma_path:
        relevant["sigma_table"] = sigma_path

    if metadata.get("module") == "tau_phase" or (module == "tau_phase" and metadata.get("energy_grid")):
        dynamic = _dynamic_tau_caption(path_obj, metadata, relevant)
        if dynamic is not None:
            return dynamic

    if metadata.get("module") == "inclination_scan" or (module == "inclination_scan" and metadata.get("scan_parameters")):
        dynamic = _dynamic_inclination_caption(path_obj, metadata, relevant)
        if dynamic is not None:
            return dynamic

    if str(metadata.get("module", module)).startswith("paper_"):
        dynamic = _dynamic_paper_caption(path_obj, metadata, relevant)
        if dynamic is not None:
            return dynamic

    if module == "inclination_scan" or "psurv_vs_inclination" in name:
        plot_type = "inclination_survival_probability"
        theta_range = _range_text(metadata.get("inclination_angles_deg"), r"^\circ")
        if theta_range:
            relevant["inclination_range"] = theta_range
        caption = (
            r"\caption{"
            r"Survival probability of ultra-high-energy neutrinos as a function of observer inclination "
            r"for geodesic propagation through the Kerr collapsar-torus background. The plotted quantities "
            r"summarize the inclination dependence of the mean survival probability, optical depth, and "
            r"observed UHE intensity, reflecting the changing column density encountered by rays crossing "
            r"the toroidal matter distribution."
            r"}"
        )
        label = r"\label{fig:uhe_survival_inclination}"
        return {
            "plot_type": plot_type,
            "caption_latex": caption,
            "label_latex": label,
            "caption_source": "auto_template",
            "relevant_parameters": relevant,
        }

    if module == "tau_phase" or "tau_phase" in name:
        energy_range = _range_text(metadata.get("energies_GeV"), r"\,\mathrm{GeV}")
        rho_range = _range_text(metadata.get("rho0_gcm3"), r"\,\mathrm{g\,cm^{-3}}")
        if energy_range:
            relevant["energy_range"] = energy_range
        if rho_range:
            relevant["density_range"] = rho_range
        if "iim_over_gbw" in name:
            plot_type = "tau_phase_ratio"
            caption = (
                r"\caption{"
                r"Ratio of the stream-mode UHE optical-depth phase maps obtained with the IIM and GBW "
                r"dipole parametrizations of the charged-current DIS neutrino--nucleon cross section. "
                r"The color scale quantifies model-dependent changes in the effective attenuation across "
                r"the neutrino-energy and torus-density grid."
                r"}"
            )
            label = r"\label{fig:tau_phase_iim_over_gbw}"
        else:
            model = "IIM" if "iim" in name else "GBW" if "gbw" in name else "DIS"
            relevant["cross_section_model"] = model
            plot_type = "tau_phase_diagram"
            caption = (
                r"\caption{"
                rf"Optical-depth phase diagram for ultra-high-energy neutrinos propagating through the "
                rf"Kerr collapsar-torus background using the {model} dipole parametrization of the "
                r"charged-current DIS neutrino--nucleon cross section. The contours trace transitions "
                r"between optically thin and attenuated propagation regimes across the neutrino-energy "
                r"and torus-density plane."
                r"}"
            )
            label = rf"\label{{fig:tau_phase_{model.lower()}}}"
        return {
            "plot_type": plot_type,
            "caption_latex": caption,
            "label_latex": label,
            "caption_source": "auto_template",
            "relevant_parameters": relevant,
        }

    if "attenuated_spectrum" in name or "spectrum" in name:
        plot_type = "uhe_spectral_attenuation"
        caption = (
            r"\caption{"
            r"Ultra-high-energy neutrino spectral attenuation after propagation through the HADROS "
            r"collapsar-torus background. The curves compare emitted and observed spectra, isolating "
            r"the energy-dependent attenuation induced by charged-current DIS opacity along the ray bundle."
            r"}"
        )
        label = rf"\label{{fig:{_label_slug(stem)}}}"
        return {
            "plot_type": plot_type,
            "caption_latex": caption,
            "label_latex": label,
            "caption_source": "auto_template",
            "relevant_parameters": relevant,
        }

    if "transfer" in name:
        plot_type = "transfer_diagnostic"
        caption = (
            r"\caption{"
            r"Transfer-function diagnostic for HADROS geodesic propagation through the collapsar-torus "
            r"background. The figure summarizes the storage or compression behavior of the transfer "
            r"representation while preserving the ray-dependent optical-depth information used by the "
            r"post-processing pipeline."
            r"}"
        )
        label = rf"\label{{fig:{_label_slug(stem)}}}"
        return {
            "plot_type": plot_type,
            "caption_latex": caption,
            "label_latex": label,
            "caption_source": "auto_template",
            "relevant_parameters": relevant,
        }

    if "adaptive" in name or "fgar" in name or "stream" in name:
        plot_type = "adaptive_rendering_diagnostic"
        caption = (
            r"\caption{"
            r"Adaptive-rendering diagnostic for HADROS stream-mode ray tracing. The figure characterizes "
            r"how feature-guided adaptive refinement allocates rays near image structures relevant to "
            r"Kerr lensing, torus emission, and neutrino-opacity gradients."
            r"}"
        )
        label = rf"\label{{fig:{_label_slug(stem)}}}"
        return {
            "plot_type": plot_type,
            "caption_latex": caption,
            "label_latex": label,
            "caption_source": "auto_template",
            "relevant_parameters": relevant,
        }

    if "kerr" in name or "photon_ring" in name:
        plot_type = "kerr_validation"
        caption = (
            r"\caption{"
            r"Kerr-ray-tracing validation diagnostic for HADROS. The plotted quantities test numerical "
            r"geodesic behavior and image-level consistency in regimes sensitive to strong-field lensing "
            r"near the black-hole photon-ring region."
            r"}"
        )
        label = rf"\label{{fig:{_label_slug(stem)}}}"
        return {
            "plot_type": plot_type,
            "caption_latex": caption,
            "label_latex": label,
            "caption_source": "auto_template",
            "relevant_parameters": relevant,
        }

    caption = (
        r"\caption{"
        rf"HADROS scientific diagnostic generated by the {module_latex} module. "
        r"The figure is included as a run-local product in the manifest; consult the associated "
        r"metadata files for the complete configuration and numerical provenance."
        r"}"
    )
    return {
        "plot_type": plot_type or "hadros_plot",
        "caption_latex": caption,
        "label_latex": rf"\label{{fig:{_label_slug(module + '_' + stem)}}}",
        "caption_source": "fallback",
        "relevant_parameters": relevant,
    }
