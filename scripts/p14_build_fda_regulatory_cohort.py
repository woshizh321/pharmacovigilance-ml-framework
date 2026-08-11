#!/usr/bin/env python3
"""Build the frozen FDA CDER NME/new-biologic regulatory cohort.

The immutable XLSX is extracted read-only with the bundled spreadsheet runtime
to a JSON value matrix. This script converts that matrix to auditable CSVs,
freezes exact active-ingredient/moiety identities, and groups repeated FDA
records by the exact FDA identity. It does not use any safety outcome data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


PROJECT = Path("/path/to/PDS")
FDA_DIR = PROJECT / "data_external/fda"
PROC = PROJECT / "data/processed/preflight_v2"
OUT = PROJECT / "preflight_v2"
MATRIX = PROC / "fda_cder_nme_1985_2025_workbook_values.json"
COHORT = OUT / "fda_cder_nme_cohort_master.csv"
RECORDS = PROC / "fda_cder_nme_1985_2025_source_records.csv"
METRICS = PROC / "fda_regulatory_cohort_metrics.json"

XLSX = FDA_DIR / "2026 Compilation_of_CDER_NME_and_New_Biologic_Approvals_1985-2025.xlsx"
DICTIONARY = FDA_DIR / "Final Draft Compilation of CDER New Molecular Entity (NME) Drug and New Biologic Approvals Data Dictionary-June 2025 update.pdf"
SOURCE_HTML = FDA_DIR / "Compilation of CDER NME source page.html"
SOURCE_URL = "https://www.fda.gov/drugs/drug-approvals-and-databases/compilation-cder-new-molecular-entity-nme-drug-and-new-biologic-approvals"
DATA_URL = "https://www.fda.gov/media/177921/download?attachment="
DICTIONARY_URL = "https://www.fda.gov/media/177920/download?attachment="
RETRIEVAL_DATE = "2026-08-09"
FAERS_CUTOFF = date(2025, 12, 31)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def identity_key(value: str) -> str:
    return clean_text(value).casefold()


def excel_date(value) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return date(1899, 12, 30) + timedelta(days=int(value))
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return datetime.strptime(text, "%m/%d/%Y").date()


def join_unique(values) -> str:
    seen = set()
    out = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return " | ".join(out)


def add_calendar_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # Regulatory anniversary convention for a 29-Feb anchor in a
        # non-leap target year: last calendar day of February.
        return value.replace(year=value.year + years, day=28)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def multi_active(term: str) -> bool:
    # FDA's source field uses commas/semicolons/"and" to enumerate active
    # ingredients. Parenthetical qualifiers alone do not trigger exclusion.
    outside_parentheses = re.sub(r"\([^)]*\)", "", term)
    return bool(re.search(r"[,;]", outside_parentheses) or re.search(r"\s+and\s+", outside_parentheses, re.I))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))["values"]
    header = matrix[0]
    raw = [dict(zip(header, row)) for row in matrix[1:]]

    records = []
    date_year_failures = 0
    for i, row in enumerate(raw, start=2):
        approval = excel_date(row["FDA Approval Date"])
        receipt = excel_date(row["FDA Receipt Date"])
        approval_year = int(row["Approval Year"])
        if approval is None or approval.year != approval_year:
            date_year_failures += 1
        active = clean_text(row["Active Ingredient/Moiety"])
        apps = [row.get(f" Application Number({j})") for j in (1, 2, 3)]
        forms = [row.get(f"Dosage Form({j})") for j in (1, 2, 3)]
        routes = [row.get(f"Route of Administration({j})") for j in (1, 2, 3)]
        record = {
            "source_row": i,
            "fda_proprietary_name": clean_text(row["Proprietary  Name"]),
            "fda_active_ingredient_moiety_raw": active,
            "fda_identity_key": identity_key(active),
            "applicant": clean_text(row["Applicant"]),
            "nda_bla": clean_text(row["NDA/BLA"]),
            "application_number": join_unique(apps),
            "dosage_form": join_unique(forms),
            "route": join_unique(routes),
            "fda_receipt_date": receipt.isoformat() if receipt else "",
            "fda_approval_date": approval.isoformat() if approval else "",
            "approval_year": approval_year,
            "abbreviated_indication": clean_text(row["Abbreviated Indication(s)"]),
            "approved_use": clean_text(row["Approved Use(s)"]),
            "priority_review": clean_text(row["Review Designation"]),
            "orphan_designation": clean_text(row["Orphan Drug Designation"]),
            "accelerated_approval": clean_text(row["Accelerated Approval"]),
            "breakthrough_therapy_designation": clean_text(row["Breakthrough Therapy Designation"]),
            "fast_track_designation": clean_text(row["Fast Track Designation"]),
            "qualified_infectious_disease_product": clean_text(row["Qualified Infectious Disease Product"]),
            "issued_priority_review_voucher": clean_text(row["Issued a Priority Review Voucher"]),
            "redeemed_priority_review_voucher": clean_text(row["Redeemed a Priority Review Voucher"]),
            "fda_comments_notes": clean_text(row["Notes"]),
            "multi_active_ingredient_flag": multi_active(active),
            "administrative_conversion_flag": bool(re.search(r"transition product|new license number", clean_text(row["Notes"]), re.I)),
        }
        records.append(record)
    write_csv(RECORDS, records)

    by_identity: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_identity[record["fda_identity_key"]].append(record)

    cohort = []
    for key, group in sorted(by_identity.items()):
        group.sort(key=lambda r: (r["fda_approval_date"], r["source_row"]))
        first = group[0]
        first_date = date.fromisoformat(first["fda_approval_date"])
        combo = any(r["multi_active_ingredient_flag"] for r in group)
        followup_ok = add_calendar_years(first_date, 3) <= FAERS_CUTOFF
        cohort.append({
            "canonical_active_moiety": first["fda_active_ingredient_moiety_raw"],
            "fda_proprietary_name": join_unique(r["fda_proprietary_name"] for r in group),
            "application_number": join_unique(r["application_number"] for r in group),
            "nda_bla": join_unique(r["nda_bla"] for r in group),
            "fda_first_approval_date": first["fda_approval_date"],
            "approval_year": first_date.year,
            "applicant": join_unique(r["applicant"] for r in group),
            "orphan_designation": first["orphan_designation"],
            "accelerated_approval": first["accelerated_approval"],
            "breakthrough_therapy_designation": first["breakthrough_therapy_designation"],
            "fast_track_designation": first["fast_track_designation"],
            "priority_review": first["priority_review"],
            "route": join_unique(r["route"] for r in group),
            "dosage_form": join_unique(r["dosage_form"] for r in group),
            "abbreviated_indication": first["abbreviated_indication"],
            "approved_use": first["approved_use"],
            "qualified_infectious_disease_product": first["qualified_infectious_disease_product"],
            "issued_priority_review_voucher": first["issued_priority_review_voucher"],
            "redeemed_priority_review_voucher": first["redeemed_priority_review_voucher"],
            "fda_comments_notes": join_unique(r["fda_comments_notes"] for r in group),
            "source_record_count": len(group),
            "all_fda_approval_dates": join_unique(r["fda_approval_date"] for r in group),
            "multiple_applications_or_products_flag": len(group) > 1 or any(" | " in r["application_number"] for r in group),
            "multi_active_ingredient_flag": combo,
            "administrative_conversion_flag": any(r["administrative_conversion_flag"] for r in group),
            "salt_ester_prodrug_review_flag": bool(re.search(
                r"\b(hydrochloride|hydrobromide|sodium|potassium|calcium|acetate|mesylate|maleate|fumarate|succinate|tartrate|citrate|phosphate|sulfate|sulphate|nitrate|ester|prodrug)\b",
                first["fda_active_ingredient_moiety_raw"], re.I,
            )),
            "three_year_followup_complete_at_faers_cutoff": followup_ok,
            "exclusion_flag": combo,
            "exclusion_reason": "MULTI_ACTIVE_COMBINATION_PRIMARY_COHORT" if combo else "",
        })
    write_csv(COHORT, cohort)

    windows = {}
    for start in (2012, 2013, 2014, 2015):
        eligible = [r for r in cohort if start <= r["approval_year"] <= 2022 and not r["exclusion_flag"]]
        windows[f"{start}-2022"] = {
            "source_active_moieties": sum(start <= r["approval_year"] <= 2022 for r in cohort),
            "primary_single_active_moieties": len(eligible),
            "excluded_combinations": sum(start <= r["approval_year"] <= 2022 and r["exclusion_flag"] for r in cohort),
            "complete_3y_followup": sum(r["three_year_followup_complete_at_faers_cutoff"] for r in eligible),
        }

    duplicate_groups = [g for g in by_identity.values() if len(g) > 1]
    manifest = {
        "source_page": SOURCE_URL,
        "retrieval_date": RETRIEVAL_DATE,
        "dataset_coverage": "1985-2025",
        "fda_description_version": "FDA page current on retrieval date; annual compilation, page updated for 1985-2025",
        "files": [
            {"role": "official_dataset", "source_url": DATA_URL, "original_filename": XLSX.name, "size_bytes": XLSX.stat().st_size, "sha256": sha256(XLSX)},
            {"role": "associated_data_dictionary", "source_url": DICTIONARY_URL, "original_filename": DICTIONARY.name, "size_bytes": DICTIONARY.stat().st_size, "sha256": sha256(DICTIONARY)},
            {"role": "source_page_archive", "source_url": SOURCE_URL, "original_filename": SOURCE_HTML.name, "size_bytes": SOURCE_HTML.stat().st_size, "sha256": sha256(SOURCE_HTML)},
        ],
        "metadata_warning": "FDA source page labels the dictionary May 2026 and dataset 1985-2025, but the linked PDF filename says June 2025 update and its body describes 1985-2024. Dataset XLSX itself contains 2025 records and is treated as authoritative.",
    }
    (FDA_DIR / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "source_records": len(records),
        "exact_fda_active_identity_groups": len(cohort),
        "duplicate_active_identity_groups": len(duplicate_groups),
        "duplicate_source_records_beyond_first": sum(len(g) - 1 for g in duplicate_groups),
        "multi_active_combination_groups": sum(r["multi_active_ingredient_flag"] for r in cohort),
        "administrative_conversion_groups": sum(r["administrative_conversion_flag"] for r in cohort),
        "salt_ester_prodrug_review_groups": sum(r["salt_ester_prodrug_review_flag"] for r in cohort),
        "date_year_consistency_failures": date_year_failures,
        "approval_year_min": min(r["approval_year"] for r in cohort),
        "approval_year_max": max(r["approval_year"] for r in cohort),
        "faers_cutoff": FAERS_CUTOFF.isoformat(),
        "candidate_windows": windows,
        "source_manifest": manifest,
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    duplicate_lines = []
    for group in duplicate_groups:
        group.sort(key=lambda r: r["fda_approval_date"])
        active_label = group[0]["fda_active_ingredient_moiety_raw"].replace("|", "\\|")
        product_label = join_unique(r["fda_proprietary_name"] for r in group).replace("|", "\\|")
        date_label = join_unique(r["fda_approval_date"] for r in group).replace("|", "\\|")
        duplicate_lines.append(
            f"| {active_label} | {len(group)} | {product_label} | {date_label} |"
        )
    window_lines = "\n".join(
        f"| {name} | {v['source_active_moieties']:,} | {v['primary_single_active_moieties']:,} | {v['excluded_combinations']:,} | {v['complete_3y_followup']:,} |"
        for name, v in windows.items()
    )
    report = f"""# 01 — FDA source and regulatory-cohort audit

## Official source freeze

- FDA source page: {SOURCE_URL}
- Retrieval date: {RETRIEVAL_DATE}
- Dataset: `{XLSX.name}` ({XLSX.stat().st_size:,} bytes; SHA256 `{sha256(XLSX)}`)
- Data dictionary: `{DICTIONARY.name}` ({DICTIONARY.stat().st_size:,} bytes; SHA256 `{sha256(DICTIONARY)}`)
- Coverage: 1985–2025, CDER Type 1/1-4 NME drugs and new therapeutic biologics; CBER-only vaccines, blood products, cell and gene therapies are outside the source compilation.

The FDA page states that the dataset reflects each application at original marketing approval and is updated annually. The page labels the dictionary “May 2026,” but the linked PDF filename says “June 2025 update” and its body still describes 1985–2024. The XLSX contains {sum(r['approval_year']==2025 for r in records):,} 2025 records and is the authoritative regulatory table; this official metadata inconsistency is retained in `SOURCE_MANIFEST.json`.

## Active-moiety/ingredient QC

| Quantity | Count |
|---|---:|
| FDA source approval records | {len(records):,} |
| Exact FDA active ingredient/moiety identity groups | {len(cohort):,} |
| Duplicate exact identity groups | {len(duplicate_groups):,} |
| Extra records beyond first approval | {sum(len(g)-1 for g in duplicate_groups):,} |
| Multi-active combination groups | {sum(r['multi_active_ingredient_flag'] for r in cohort):,} |
| Administrative conversion/new-license groups | {sum(r['administrative_conversion_flag'] for r in cohort):,} |
| Salt/ester/prodrug manual-review flags | {sum(r['salt_ester_prodrug_review_flag'] for r in cohort):,} |
| FDA approval-date/year inconsistencies | {date_year_failures:,} |

Exact duplicate active identities are frozen to the earliest original FDA approval date while all products, applications and dates remain in delimited audit fields.

| Repeated active ingredient/moiety | Source records | Products | Approval dates |
|---|---:|---|---|
{chr(10).join(duplicate_lines)}

All multi-active products are retained in the cohort master but excluded from the primary single-active-moiety cohort. No component of a combination is credited with the combination product's approval.

## Candidate approval windows

| Window | Source active identities | Primary single-active identities | Excluded combinations | Complete exact 3-year FAERS follow-up |
|---|---:|---:|---:|---:|
{window_lines}

FAERS cutoff is {FAERS_CUTOFF.isoformat()}; every retained identity through 2022 has a complete exact three-calendar-year follow-up window.
"""
    (OUT / "01_fda_source_and_cohort_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
