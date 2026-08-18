from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TAB = ROOT / "tables"
DOC = ROOT / "docs" / "PUBLICATION_VISUAL_LOCK_v1.md"
ANALYSIS_BEFORE = Path("/tmp/pvml_command20_analysis_before.sha256")
UNCHANGED_FIGURES_BEFORE = Path("/tmp/pvml_command20_unchanged_figures_before.sha256")
TABLES_BEFORE = Path("/tmp/pvml_command20_tables_before.sha256")

FIGURE_BASES = [
    "Figure1_study_design",
    "Figure2_coverage",
    "Figure3_prediction",
    "Figure4_transport_interpretation",
    "Figure5_jader_robustness",
]
TABLE_BASES = [
    "Table1_cohort_characteristics",
    "Table2_model_performance",
    "Table3_jader_replication",
]
FIGURE_EXTENSIONS = [".svg", ".pdf", ".tiff", ".png"]
TABLE_EXTENSIONS = [".xlsx", ".csv"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(paths: list[Path]) -> str:
    return "".join(f"{digest(path)}  {path.relative_to(ROOT)}\n" for path in sorted(paths))


def analysis_files() -> list[Path]:
    return sorted(path for path in (ROOT / "analysis").rglob("*") if path.is_file())


def unchanged_figure_files() -> list[Path]:
    keep = {FIGURE_BASES[i] for i in (0, 2, 4)}
    return sorted(
        path for path in (FIG / "main").iterdir()
        if path.is_file() and path.stem in keep and path.suffix in FIGURE_EXTENSIONS
    )


def table_files() -> list[Path]:
    return [TAB / "main" / f"{base}{ext}" for base in TABLE_BASES for ext in TABLE_EXTENSIONS]


def frame_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True), right.reset_index(drop=True),
            check_dtype=False, check_like=False,
        )
        return True
    except AssertionError:
        return False


def markdown_checksum_table(paths: list[Path]) -> list[str]:
    rows = ["| Final file | SHA-256 |", "|---|---|"]
    rows.extend(
        f"| `{path.relative_to(ROOT)}` | `{digest(path)}` |"
        for path in paths
    )
    return rows


def main() -> None:
    checks: dict[str, bool] = {}

    checks["01_analysis_directory_unchanged"] = (
        ANALYSIS_BEFORE.exists() and inventory(analysis_files()) == ANALYSIS_BEFORE.read_text(encoding="utf-8")
    )
    checks["02_figures_1_3_5_byte_unchanged"] = (
        UNCHANGED_FIGURES_BEFORE.exists()
        and inventory(unchanged_figure_files()) == UNCHANGED_FIGURES_BEFORE.read_text(encoding="utf-8")
    )
    checks["03_tables_1_2_3_byte_unchanged"] = (
        TABLES_BEFORE.exists() and inventory(table_files()) == TABLES_BEFORE.read_text(encoding="utf-8")
    )

    cov = pd.read_csv(ROOT / "analysis/section4_holdout/12_full_2012_2022_coverage.csv")
    periods = cov[cov["scope"].isin(["DEVELOPMENT_2012_2018", "TEMPORAL_HOLDOUT_2019_2022"])]
    source_c = pd.read_csv(FIG / "source_data/Figure2_panelC_drug_distribution_summary.csv")
    cols_c = [
        "scope", "active_moieties", "median_drug_coverage_percent",
        "drug_coverage_q1_percent", "drug_coverage_q3_percent",
    ]
    svg2 = (FIG / "main/Figure2_coverage.svg").read_text(encoding="utf-8")
    captions = (FIG / "FIGURE_CAPTIONS_v1.md").read_text(encoding="utf-8")
    checks["04_figure2_panel_c_frozen_source_and_0_35_axis"] = (
        frame_equal(source_c[cols_c], periods[cols_c])
        and ">35</text>" in svg2
        and ">0</text>" in svg2
        and "0–35% axis" in captions
        and "do not display the full drug-level range" in captions
    )
    checks["05_figure2_canonical_and_vb_assets_identical"] = all(
        digest(FIG / "main" / f"Figure2_coverage{ext}")
        == digest(FIG / "main" / f"Figure2_coverage_vB{ext}")
        for ext in FIGURE_EXTENSIONS
    )

    dep = pd.read_csv(
        ROOT / "analysis/section5_interpretation/12_figure_interpretation_source_data/panel_c_supported_feature_dependence.csv"
    )
    dep_features = [
        "pair_reporting_trial_fraction",
        "pair_median_row_ae_proportion",
        "pair_n_serious_arms",
    ]
    dep = dep[dep["conceptual_feature"].isin(dep_features)]
    dep_source = pd.read_csv(FIG / "source_data/Figure4_panelD_OOF_dependence.csv")
    svg4 = (FIG / "main/Figure4_transport_interpretation.svg").read_text(encoding="utf-8")
    guide_sentence = (
        "Connected lines are descriptive guides across prespecified quantile-bin summaries "
        "and do not represent fitted functional forms."
    )
    checks["06_figure4_panel_d_frozen_source_and_labeling"] = (
        frame_equal(dep_source, dep)
        and "OOF quantile bin" in svg4
        and "median value shown" in svg4
        and "Frozen bin median" not in svg4
        and guide_sentence in captions
    )

    figure_paths = [FIG / "main" / f"{base}{ext}" for base in FIGURE_BASES for ext in FIGURE_EXTENSIONS]
    checks["07_final_main_figure_assets_complete"] = all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
    checks["08_final_main_table_assets_complete"] = all(path.exists() and path.stat().st_size > 0 for path in table_files())
    checks["09_vector_and_600dpi_assets_valid"] = all(
        (FIG / "main" / f"{base}.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")
        and (FIG / "main" / f"{base}.pdf").read_bytes()[:4] == b"%PDF"
        and min(Image.open(FIG / "main" / f"{base}.tiff").info.get("dpi", (0, 0))) >= 599
        for base in FIGURE_BASES
    )

    visual_manifest = pd.read_csv(FIG / "VISUALIZATION_MANIFEST_v1.csv")
    table_manifest = pd.read_csv(TAB / "TABLE_MANIFEST_v1.csv")
    required_figures = {"Figure1", "Figure2", "Figure3", "Figure4", "Figure5"}
    required_tables = {"Table 1", "Table 2A", "Table 2B", "Table 3"}
    checks["10_source_manifests_complete_and_pass"] = (
        visual_manifest["numerical_QC_status"].eq("PASS").all()
        and table_manifest["numerical_QC_result"].eq("PASS").all()
        and required_figures.issubset(set(visual_manifest["figure_id"]))
        and required_tables.issubset(set(table_manifest["table"]))
    )
    row_2c = visual_manifest[(visual_manifest["figure_id"] == "Figure2") & (visual_manifest["panel"] == "C")]
    row_4d = visual_manifest[(visual_manifest["figure_id"] == "Figure4") & (visual_manifest["panel"] == "D")]
    checks["11_manifest_records_final_visual_transformations"] = (
        len(row_2c) == 1
        and "0–35% axis" in row_2c.iloc[0]["transformation_for_display"]
        and len(row_4d) == 1
        and "quantile-bin medians" in row_4d.iloc[0]["transformation_for_display"]
    )

    prior_table_qc = json.loads((ROOT / "TABLE_QC.json").read_text(encoding="utf-8"))
    checks["12_prior_final_table_qc_pass"] = prior_table_qc.get("status") == "PASS"
    checks = {name: bool(value) for name, value in checks.items()}

    status = "PASS" if all(checks.values()) else "FAIL"
    ancillary = [
        FIG / "FIGURE_CAPTIONS_v1.md",
        TAB / "TABLE_FOOTNOTES_v1.md",
        FIG / "VISUALIZATION_MANIFEST_v1.csv",
        TAB / "TABLE_MANIFEST_v1.csv",
    ]
    lines = [
        "# Publication Visual Lock v1",
        "",
        f"Status: **{status}**",
        "",
        "Lock date: 2026-08-10",
        "",
        "## Scope and scientific integrity",
        "",
        "Figures 1–5 and Tables 1–3 are frozen for publication use. All visualizations derive only from frozen analysis outputs enumerated in the source-data manifests. Command 20 performed display-only rendering, caption editing, provenance verification, and checksum generation. No statistic, confidence interval, bootstrap, model, prediction, SHAP value, calibration estimate, feature rank, endpoint result, or JADER result was recomputed.",
        "",
        "Figure 2 Panel C uses a 0–35% y-axis and displays frozen median/IQR summaries, not the full drug-level range. Figure 4 Panel D labels the x-axis as OOF quantile bins with median values shown; connected lines are descriptive guides, not fitted functional forms. Figures 1, 3, and 5 and Tables 1–3 remained byte-for-byte unchanged during Command 20.",
        "",
        "## Final main figures and checksums",
        "",
    ]
    lines.extend(markdown_checksum_table(figure_paths))
    lines.extend([
        "",
        "The retained `Figure2_coverage_vB` files are byte-identical aliases of the canonical `Figure2_coverage` files in all four formats.",
        "",
        "## Final main tables and checksums",
        "",
    ])
    lines.extend(markdown_checksum_table(table_files()))
    lines.extend([
        "",
        "## Captions, footnotes, and source-data manifests",
        "",
    ])
    lines.extend(markdown_checksum_table(ancillary))
    lines.extend([
        "",
        "The visualization manifest provides panel-level frozen-source provenance; the table manifest provides table/block-level frozen-source provenance. The caption and footnote files are the final publication text companions for these locked assets.",
        "",
        "## Visual/source-provenance QC",
        "",
    ])
    lines.extend(f"- {'PASS' if ok else 'FAIL'} — {name}" for name, ok in checks.items())
    lines.extend([
        "",
        "## Lock rule",
        "",
        "Any later modification to a locked figure, table, caption, footnote, source-data manifest, filename, or checksum requires an explicit new command, a versioned replacement lock, and renewed visual/source-provenance QC. Locked scientific values must never be silently overwritten.",
    ])
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "status": status,
        "scope": "COMMAND_20_FINAL_VISUAL_LOCK",
        "scientific_analysis_authorized": False,
        "scientific_statistics_recomputed": False,
        "confidence_intervals_recomputed": False,
        "bootstrap_rerun": False,
        "models_refit": False,
        "predictions_regenerated": False,
        "SHAP_recalculated": False,
        "calibration_recomputed": False,
        "JADER_results_recomputed": False,
        "figure2_panel_c_axis": "0–35%",
        "figure4_panel_d_label": "OOF quantile bin (median value shown)",
        "qc_gates": checks,
        "lock_document": str(DOC.relative_to(ROOT)),
    }
    (ROOT / "PUBLICATION_VISUAL_LOCK_QC.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
