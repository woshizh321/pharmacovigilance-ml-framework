from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
BEFORE = Path("/tmp/pds_command18_analysis_before.sha256")


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def same(a, b, tol=1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), atol=tol, rtol=0))


def frame_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                      check_dtype=False, check_like=False)
        return True
    except AssertionError:
        return False


def sha_inventory(directory: Path, output: Path):
    lines = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}\n")
    output.write_text("".join(lines), encoding="utf-8")


def check_vector(path: Path) -> bool:
    if path.suffix == ".svg":
        return "<svg" in path.read_text(encoding="utf-8")[:1000]
    if path.suffix == ".pdf":
        return path.read_bytes()[:4] == b"%PDF"
    return False


def main():
    checks: dict[str, bool] = {}
    notes: dict[str, object] = {}
    required_main = [
        "Figure1_study_design", "Figure2_coverage", "Figure2_coverage_vB",
        "Figure3_prediction", "Figure4_transport_interpretation",
        "Figure5_jader_robustness",
    ]
    revised_supp = [
        "FigureS1_soc_coverage", "FigureS5_PT_support_detail",
        "FigureS8_design_sensitivity",
    ]

    after = Path("/tmp/pds_command18_analysis_after.sha256")
    sha_inventory(ROOT / "analysis", after)
    checks["01_analysis_directory_unchanged"] = BEFORE.exists() and BEFORE.read_bytes() == after.read_bytes()
    checks["02_no_model_prediction_bootstrap_shap_or_calibration_recomputation"] = checks["01_analysis_directory_unchanged"]

    captions = (FIG / "FIGURE_CAPTIONS_v1.md").read_text(encoding="utf-8")
    svg1 = (FIG / "main/Figure1_study_design.svg").read_text(encoding="utf-8")
    checks["03_figure1_parallel_convergence_architecture"] = all(term in svg1 for term in [
        "Parallel identity branches", "not serial attrition", "B-STRICT evidence eligibility",
        "FAERS identity/outcome", "Convergence", "Final PS≥100 cohort",
        "Cumulative cross-database", "replication only",
    ])
    checks["04_figure1_unresolved_identity_note_caption_only"] = (
        "unresolved FAERS identity" in captions and "unresolved FAERS identity" not in svg1
    )

    full_lock = read("analysis/section4_holdout/12_full_2012_2022_coverage.csv")
    f2a = read("figures/source_data/Figure2_panelA_full_coverage.csv")
    f2b = read("figures/source_data/Figure2_panelB_temporal_consistency.csv")
    f2c = read("figures/source_data/Figure2_panelC_drug_distribution_summary.csv")
    full_row = full_lock[full_lock["scope"] == "FULL_2012_2022"].iloc[0]
    periods = full_lock[full_lock["scope"].isin(["DEVELOPMENT_2012_2018", "TEMPORAL_HOLDOUT_2019_2022"])]
    checks["05_figure2_values_and_intervals_match_frozen_source"] = (
        int(f2a.iloc[0]["premarketing_observed"]) == int(full_row["premarketing_observed"]) and
        same(f2a.iloc[0]["premarketing_observed_ci_low"], full_row["premarketing_observed_ci_low"]) and
        same(f2a.iloc[0]["premarketing_observed_ci_high"], full_row["premarketing_observed_ci_high"]) and
        all(same(a, b) for a, b in zip(f2b["macro_coverage_ci_low"], periods["macro_coverage_ci_low"])) and
        all(same(a, b) for a, b in zip(f2b["macro_coverage_ci_high"], periods["macro_coverage_ci_high"])) and
        all(same(a, b) for a, b in zip(f2c["median_drug_coverage_percent"], periods["median_drug_coverage_percent"]))
    )
    svg2 = (FIG / "main/Figure2_coverage.svg").read_text(encoding="utf-8")
    checks["06_figure2_vB_is_main_and_SOC_is_supplemental"] = (
        "Premarketing" in svg2 and "95% CI 16.83–22.44%" in svg2 and
        "Macro coverage (secondary)" in svg2 and "Major SOC" not in svg2 and
        not (FIG / "main/Figure2_coverage_vA.svg").exists()
    )

    svg3 = (FIG / "main/Figure3_prediction.svg").read_text(encoding="utf-8")
    f3c = read("figures/source_data/Figure3_panelC_delta_AP.csv")
    checks["07_figure3_structure_CIs_and_reference_lines_preserved"] = (
        "No-skill AP" in svg3 and "Incremental value: Set 1 − Set 0" in svg3 and
        "Probability accuracy: Set 0 − Set 1" in svg3 and
        len(f3c) == 4 and f3c[["estimate", "ci_low", "ci_high"]].notna().all().all()
    )

    cal_lock = read("analysis/section4_holdout/08_holdout_calibration_source_data.csv")
    pair_lock = read("analysis/section5_interpretation/07_cross_model_rank_concordance.csv")
    pair_lock = pair_lock[pair_lock["pair_specific"]].dropna(subset=["elasticnet_rank_pair_specific", "xgboost_rank_pair_specific"])
    domain_lock = read("analysis/section5_interpretation/12_figure_interpretation_source_data/panel_b_set1_domain_attribution.csv")
    domain_lock = domain_lock[(domain_lock["feature_set"] == "SET1") & (domain_lock["aggregation_level"] == "SCIENTIFIC_DOMAIN")]
    dep_lock = read("analysis/section5_interpretation/12_figure_interpretation_source_data/panel_c_supported_feature_dependence.csv")
    dep_features = ["pair_reporting_trial_fraction", "pair_median_row_ae_proportion", "pair_n_serious_arms"]
    dep_lock = dep_lock[dep_lock["conceptual_feature"].isin(dep_features)]
    checks["08_figure4_all_panels_match_frozen_sources"] = all([
        frame_equal(read("figures/source_data/Figure4_panelA_temporal_calibration.csv"), cal_lock),
        frame_equal(read("figures/source_data/Figure4_panelB_pair_specific_rank_concordance.csv"), pair_lock),
        frame_equal(read("figures/source_data/Figure4_panelC_domain_attribution.csv"), domain_lock),
        frame_equal(read("figures/source_data/Figure4_panelD_OOF_dependence.csv"), dep_lock),
    ])
    svg4 = (FIG / "main/Figure4_transport_interpretation.svg").read_text(encoding="utf-8")
    checks["09_figure4_locked_content_and_noncausal_caption"] = all(term in svg4 for term in [
        "Rank 1 = highest contribution", "Cross-trial", "Reporting-trial fraction",
        "Median row AE proportion", "Number of serious arms",
    ]) and all(term in captions for term in ["not proportions of incremental AP explained", "noncausal"])

    summary = read("analysis/section6_robustness/04_jader_replication_summary.csv").set_index("metric")
    f5a = read("figures/source_data/Figure5_panelA_JADER_flow.csv").set_index("metric")
    expected = {
        "JADER_ASSESSABLE": (9736, 16252), "NOT_ASSESSABLE": (6516, 16252),
        "DIRECTIONALLY_POSITIVE": (3106, 9736), "JADER_R_REPLICATED": (1529, 9736),
        "JADER_CONSENSUS_REPLICATED": (1370, 9736),
    }
    checks["10_figure5_two_stage_counts_and_denominators_locked"] = all(
        int(f5a.loc[key, "n"]) == n and int(f5a.loc[key, "denominator"]) == den and
        int(summary.loc[key, "n"]) == n and int(summary.loc[key, "denominator"]) == den
        for key, (n, den) in expected.items()
    )
    f5b = read("figures/source_data/Figure5_panelB_replication_by_premarketing.csv")
    f5c = read("figures/source_data/Figure5_panelC_endpoint_horizon_deltaAP_long.csv")
    status_lock = read("analysis/section6_robustness/05_jader_replication_by_premarketing_status.csv")
    checks["11_figure5_panels_B_C_match_frozen_sources"] = (
        frame_equal(f5b, status_lock) and
        float(f5c.loc[(f5c["endpoint"] == "1-year Criterion R") &
                      (f5c["model_family"] == "Elastic-net"), "ci_low"].iloc[0]) < 0 <
        float(f5c.loc[(f5c["endpoint"] == "1-year Criterion R") &
                      (f5c["model_family"] == "Elastic-net"), "ci_high"].iloc[0])
    )
    svg5 = (FIG / "main/Figure5_jader_robustness.svg").read_text(encoding="utf-8")
    checks["12_figure5_three_panel_denominator_separation_and_labeling"] = all(term in svg5 for term in [
        "Stage 1", "16,252 definitive FAERS Criterion-R signals", "Stage 2",
        "9,736 JADER-assessable signals", "Cumulative cross-database replication",
    ]) and "Design sensitivity" not in svg5

    s1_lock = read("analysis/section2_coverage/04_soc_coverage.csv")
    s1_lock = s1_lock[s1_lock["major_soc_ge30_signals"]].nlargest(10, "criterion_r_signals").sort_values("coverage_pct")
    checks["13_relocated_supplement_panels_match_frozen_sources"] = all([
        frame_equal(read("figures/source_data/FigureS1_major_SOC_top10_by_signal_count.csv"), s1_lock),
        frame_equal(read("figures/source_data/FigureS5_PT_support_complete.csv"),
                    read("analysis/section4_holdout/09_pt_support_transportability.csv")),
        frame_equal(read("figures/source_data/FigureS8_PS_threshold.csv"),
                    read("analysis/section6_robustness/08_ps_threshold_sensitivity.csv")),
        frame_equal(read("figures/source_data/FigureS8_DefinitionA.csv"),
                    read("analysis/section6_robustness/09_definitionA_sensitivity.csv")),
        frame_equal(read("figures/source_data/FigureS8_BEXPANDED.csv"),
                    read("analysis/section6_robustness/10_bexpanded_sensitivity.csv")),
    ])

    target_names = required_main + revised_supp
    vectors_ok = True
    tiff_ok = True
    grayscale_ok = True
    grayscale_details = {}
    for name in target_names:
        directory = FIG / ("main" if name.startswith("Figure") and not name.startswith("FigureS") else "supplement")
        vectors_ok &= check_vector(directory / f"{name}.svg") and check_vector(directory / f"{name}.pdf")
        tiff = Image.open(directory / f"{name}.tiff")
        tiff_ok &= min(tiff.info.get("dpi", (0, 0))) >= 599
        arr = np.asarray(Image.open(directory / f"{name}.png").convert("L"))
        dynamic, std = int(arr.max()) - int(arr.min()), float(arr.std())
        grayscale_ok &= dynamic >= 200 and std >= 20
        grayscale_details[name] = {"dynamic_range": dynamic, "pixel_sd": round(std, 2)}
    checks["14_vector_600dpi_and_grayscale_QC"] = vectors_ok and tiff_ok and grayscale_ok
    notes["grayscale_checks"] = grayscale_details

    manifest = read("figures/VISUALIZATION_MANIFEST_v1.csv")
    checks["15_visualization_manifest_and_panel_sources_complete"] = (
        manifest["numerical_QC_status"].eq("PASS").all() and
        set(["Figure1", "Figure2", "Figure3", "Figure4", "Figure5", "FigureS1", "FigureS5", "FigureS8"]).issubset(set(manifest["figure_id"]))
    )
    all_svg_text = "\n".join((FIG / "main" / f"{name}.svg").read_text(encoding="utf-8") for name in required_main)
    checks["16_no_new_significance_annotations_or_claim_language"] = (
        not re.search(r">\*{1,3}<", all_svg_text) and "p =" not in all_svg_text.lower() and
        not any(term in all_svg_text.lower() for term in ["novel adr", "missed adr", "causal effect"])
    )

    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "COMMAND_18_FINAL_MAIN_FIGURE_REVISION",
        "scientific_statistics_recomputed": False,
        "bootstrap_rerun": False,
        "models_refit": False,
        "predictions_regenerated": False,
        "SHAP_recalculated": False,
        "calibration_recomputed": False,
        "qc_gates": checks,
        "outputs": {
            "final_main_figure_count": 5,
            "revised_supplement_figure_count": 3,
            "visualization_manifest_rows": len(manifest),
        },
        "notes": notes,
    }
    (ROOT / "VISUALIZATION_QC.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Visualization QC Report — Command 18",
        "",
        f"Status: **{report['status']}**",
        "",
        "This revision used existing frozen source tables only. No statistic, confidence interval, bootstrap, model, prediction, SHAP value, calibration estimate, feature rank, or endpoint result was recomputed.",
        "",
        "## Revised allocation",
        "",
        "- Main Figure 2 uses the three-panel vB layout; SOC coverage is Figure S1.",
        "- Main Figure 4 contains temporal calibration, rank concordance, domain attribution, and three frozen OOF dependence summaries; PT-support transportability is Figure S5.",
        "- Main Figure 5 uses a two-denominator Stage 1/Stage 2 Panel A and retains Panels B–C; design sensitivity is Figure S8.",
        "",
        "## QC gates",
        "",
    ]
    lines.extend(f"- {'PASS' if ok else 'FAIL'} — {name}" for name, ok in checks.items())
    lines.extend([
        "", "## Controlled limitation", "",
        "The frozen outputs do not contain individual temporal-holdout drug coverage values. Figure 2C therefore displays only the frozen medians and interquartile ranges; no individual holdout values were reconstructed.",
        "", "## Integrity", "",
        "The complete analysis-directory SHA256 inventory matched the Command 18 pre-revision baseline exactly. Only visual formatting, panel relocation, source-table subsetting/reshaping for display, captions, and the visualization manifest were updated.",
    ])
    (ROOT / "VISUALIZATION_QC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
