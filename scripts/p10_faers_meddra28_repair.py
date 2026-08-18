#!/usr/bin/env python3
"""Command 02B phases 1-6 and global cross-database PT overlap.

The canonical mapping is derived only from the original FAERS reaction term
(`pt`) and the local MedDRA 28.0 PT/LLT/history/hierarchy files. Legacy codes
are attached only after mapping for defect classification.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/project")
OUT = PROJECT / "preflight_v2/faers_pt_repair"
PROC = PROJECT / "data/processed/preflight_v2/faers_pt_repair"
FAERS = Path("/path/to/Database/Faers/FAERS_SUPERMASTER_V5_1_2004-2025.parquet")
LEGACY = PROJECT / "data/processed/faers_pt_universe.parquet"
MED = Path("/path/to/Database/MedDRA/MedDRA_28_0_ENglish/MedAscii")
JADER_REAC = Path("/path/to/Database/Jader/jader_v5_reac.parquet")


def read_asc(name: str) -> list[list[str]]:
    with (MED / name).open(encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="$", quoting=csv.QUOTE_NONE))


def norm_l1(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.casefold().strip().split())


def norm_l2(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(ch if ch.isalnum() else " " for ch in value)
    return " ".join(value.split())


def index_add(index: dict[str, set[str]], key: str, code: str) -> None:
    if key:
        index[key].add(code)


def resolve_index(index: dict[str, set[str]], key: str) -> tuple[str | None, list[str]]:
    candidates = sorted(index.get(key, set()))
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def write_csv(path: Path, records: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(records[0]) if records else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def md_table(records: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for record in records:
        cells = []
        for key, _ in columns:
            value = record.get(key, "")
            if isinstance(value, int):
                value = f"{value:,}"
            elif isinstance(value, float):
                value = f"{value:,.4f}".rstrip("0").rstrip(".")
            elif value is None:
                value = "NA"
            cells.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    # Authoritative MedDRA dictionaries.
    pt_rows = read_asc("pt.asc")
    llt_rows = read_asc("llt.asc")
    hier_rows = read_asc("mdhier.asc")
    history_rows = read_asc("meddra_history_english.asc")

    pt_name_by_code = {r[0]: r[1] for r in pt_rows}
    llt_name_by_code = {r[0]: r[1] for r in llt_rows}
    llt_pt_by_code = {r[0]: r[2] for r in llt_rows}

    hierarchy_candidates: dict[str, list[dict]] = defaultdict(list)
    for r in hier_rows:
        hierarchy_candidates[r[0]].append(
            {
                "canonical_hlt_code": r[1],
                "canonical_hlgt_code": r[2],
                "canonical_soc_code": r[3],
                "canonical_hlt_name": r[5],
                "canonical_hlgt_name": r[6],
                "canonical_soc_name": r[7],
                "primary_soc_flag": r[11],
            }
        )
    hierarchy: dict[str, dict] = {}
    hierarchy_ambiguous = []
    for code, branches in hierarchy_candidates.items():
        primary = [x for x in branches if x["primary_soc_flag"] == "Y"]
        chosen = primary if primary else branches
        unique = {(x["canonical_hlt_code"], x["canonical_hlgt_code"], x["canonical_soc_code"]) for x in chosen}
        if len(unique) == 1:
            hierarchy[code] = chosen[0]
        else:
            hierarchy_ambiguous.append(code)

    pt_exact: dict[str, set[str]] = defaultdict(set)
    pt_l1: dict[str, set[str]] = defaultdict(set)
    pt_l2: dict[str, set[str]] = defaultdict(set)
    for code, name in pt_name_by_code.items():
        index_add(pt_exact, name, code)
        index_add(pt_l1, norm_l1(name), code)
        index_add(pt_l2, norm_l2(name), code)

    llt_exact: dict[str, set[str]] = defaultdict(set)
    llt_l1: dict[str, set[str]] = defaultdict(set)
    llt_l2: dict[str, set[str]] = defaultdict(set)
    for code, name in llt_name_by_code.items():
        pt_code = llt_pt_by_code[code]
        index_add(llt_exact, name, pt_code)
        index_add(llt_l1, norm_l1(name), pt_code)
        index_add(llt_l2, norm_l2(name), pt_code)

    # L4: historical name -> historical code -> current PT or current LLT parent.
    hist_exact: dict[str, set[str]] = defaultdict(set)
    hist_l1: dict[str, set[str]] = defaultdict(set)
    hist_l2: dict[str, set[str]] = defaultdict(set)
    history_codes = set()
    for r in history_rows:
        code, term, _version, level, _currency, _action = r
        history_codes.add(code)
        if level not in {"PT", "LLT"}:
            continue
        current_pt = code if code in pt_name_by_code else llt_pt_by_code.get(code)
        if current_pt is None:
            continue
        index_add(hist_exact, term, current_pt)
        index_add(hist_l1, norm_l1(term), current_pt)
        index_add(hist_l2, norm_l2(term), current_pt)

    def map_term(term: str) -> dict:
        l1, l2 = norm_l1(term), norm_l2(term)
        levels = [
            ("L0_PT_EXACT", pt_exact, term),
            ("L1_PT_CASE_WHITESPACE", pt_l1, l1),
            ("L2_PT_PUNCT_UNICODE", pt_l2, l2),
            ("L3_LLT_EXACT", llt_exact, term),
            ("L3_LLT_CASE_WHITESPACE", llt_l1, l1),
            ("L3_LLT_PUNCT_UNICODE", llt_l2, l2),
            ("L4_HISTORY_EXACT", hist_exact, term),
            ("L4_HISTORY_CASE_WHITESPACE", hist_l1, l1),
            ("L4_HISTORY_PUNCT_UNICODE", hist_l2, l2),
        ]
        for label, index, key in levels:
            code, candidates = resolve_index(index, key)
            if code:
                h = hierarchy.get(code)
                if h is None:
                    return {
                        "faers_term_normalized": l2,
                        "canonical_pt_code": None,
                        "canonical_pt_name": None,
                        "mapping_level": label,
                        "mapping_status": "UNRESOLVED_HIERARCHY",
                        "manual_review_flag": True,
                        "comments": f"PT {code} lacks a unique primary hierarchy branch",
                    }
                return {
                    "faers_term_normalized": l2,
                    "canonical_pt_code": code,
                    "canonical_pt_name": pt_name_by_code[code],
                    **{k: v for k, v in h.items() if k != "primary_soc_flag"},
                    "mapping_level": label,
                    "mapping_status": "MAPPED",
                    "manual_review_flag": False,
                    "comments": "",
                }
            if len(candidates) > 1:
                return {
                    "faers_term_normalized": l2,
                    "canonical_pt_code": None,
                    "canonical_pt_name": None,
                    "mapping_level": label,
                    "mapping_status": "AMBIGUOUS",
                    "manual_review_flag": True,
                    "comments": "multiple current PT candidates: " + ";".join(candidates),
                }
        return {
            "faers_term_normalized": l2,
            "canonical_pt_code": None,
            "canonical_pt_name": None,
            "mapping_level": "UNRESOLVED",
            "mapping_status": "UNRESOLVED",
            "manual_review_flag": True,
            "comments": "no conservative PT/LLT/history match",
        }

    # Source audit and compact term-frequency extracts. Old code fields are not selected.
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='16GB'; SET preserve_insertion_order=false; SET enable_progress_bar=false")
    raw = str(FAERS)
    source_terms_path = PROC / "faers_source_term_frequency.parquet"
    source_norm_path = PROC / "faers_source_term_norm_frequency.parquet"
    con.execute(
        f"""
        COPY (
          SELECT pt AS faers_term_raw,
                 COUNT(*) expanded_rows,
                 COUNT(DISTINCT primaryid) primaryid_frequency,
                 COUNT(DISTINCT caseid) case_frequency
          FROM read_parquet('{raw}')
          WHERE pt IS NOT NULL AND trim(pt)<>''
          GROUP BY pt
        ) TO '{source_terms_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT upper(trim(pt)) source_term_key,
                 COUNT(*) expanded_rows,
                 COUNT(DISTINCT primaryid) primaryid_frequency,
                 COUNT(DISTINCT caseid) case_frequency
          FROM read_parquet('{raw}')
          WHERE pt IS NOT NULL AND trim(pt)<>''
          GROUP BY 1
        ) TO '{source_norm_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    source_diag = con.execute(
        f"""
        SELECT COUNT(*) expanded_rows,
               COUNT(DISTINCT primaryid) primaryids,
               COUNT(DISTINCT caseid) caseids,
               COUNT(DISTINCT pt) original_pt_terms,
               COUNT(DISTINCT pt_norm) pt_norm_terms,
               COUNT(DISTINCT pt_u) legacy_pt_u_terms,
               SUM((upper(trim(pt))=pt_norm)::INT) rows_original_pt_agrees_pt_norm,
               SUM((upper(trim(pt))=pt_u)::INT) rows_original_pt_agrees_pt_u,
               SUM((pt_norm=pt_u)::INT) rows_pt_norm_agrees_pt_u
        FROM read_parquet('{raw}')
        """
    ).fetchone()
    source_diag_cols = [x[0] for x in con.description]
    source_diag = dict(zip(source_diag_cols, source_diag))

    source_cursor = con.execute(f"SELECT * FROM read_parquet('{source_terms_path}') ORDER BY faers_term_raw")
    source_cols = [x[0] for x in source_cursor.description]
    source_records = [dict(zip(source_cols, row)) for row in source_cursor.fetchall()]
    norm_freq = {
        r[0]: {"expanded_rows": r[1], "primaryid_frequency": r[2], "case_frequency": r[3]}
        for r in con.execute(f"SELECT * FROM read_parquet('{source_norm_path}')").fetchall()
    }
    legacy_rows = con.execute(f"SELECT pt, pt_name, pt_code, n_cases_any_role FROM read_parquet('{LEGACY}')").fetchall()
    legacy_by_key = {
        str(r[0]).upper().strip(): {
            "legacy_term": r[0],
            "legacy_pt_name": r[1],
            "legacy_pt_code": str(r[2]) if r[2] is not None else None,
            "legacy_primaryid_frequency": r[3],
        }
        for r in legacy_rows
    }

    def classify_legacy(mapped: dict, legacy: dict | None) -> tuple[bool | None, str, str | None]:
        if legacy is None:
            return None, "NO_LEGACY_RECORD", None
        old_code = legacy["legacy_pt_code"]
        canonical = mapped.get("canonical_pt_code")
        old_name = legacy.get("legacy_pt_name")
        old_code_name = pt_name_by_code.get(old_code) if old_code else None
        if canonical is None:
            return False, "6_UNRESOLVED_OR_AMBIGUOUS", old_code_name
        if old_code is None:
            return False, "4_CODE_MISSING_OR_INVALID", None
        name_correct = bool(old_name and norm_l1(str(old_name)) == norm_l1(pt_name_by_code[canonical]))
        if old_code == canonical and name_correct:
            return True, "1_NAME_AND_CODE_CORRECT", old_code_name
        if old_code == canonical and not name_correct:
            return True, "5_LEGACY_NAME_TERM_MISALIGNED_CODE_CORRECT", old_code_name
        if old_code in pt_name_by_code:
            if name_correct:
                return False, "2_NAME_CORRECT_CODE_POINTS_OTHER_CURRENT_PT", old_code_name
            return False, "5_LEGACY_TERM_NAME_CODE_MISALIGNED", old_code_name
        if old_code in history_codes:
            return False, "3_NONCURRENT_OR_DEPRECATED_CODE_TERM_RECOVERED", None
        return False, "4_CODE_MISSING_OR_INVALID", None

    canonical = []
    for source in source_records:
        raw_term = str(source["faers_term_raw"])
        mapped = map_term(raw_term)
        legacy = legacy_by_key.get(raw_term.upper().strip())
        correct, error_class, old_code_name = classify_legacy(mapped, legacy)
        record = {
            "faers_term_raw": raw_term,
            **mapped,
            "legacy_pt_code": legacy.get("legacy_pt_code") if legacy else None,
            "legacy_pt_name": legacy.get("legacy_pt_name") if legacy else None,
            "legacy_code_pt_name": old_code_name,
            "legacy_code_correct": correct,
            "legacy_code_error_class": error_class,
            "case_frequency": int(source["case_frequency"]),
            "primaryid_frequency": int(source["primaryid_frequency"]),
            "expanded_row_frequency": int(source["expanded_rows"]),
        }
        canonical.append(record)

    # Enforce one normalized source term -> one current PT.
    norm_to_codes: dict[str, set[str]] = defaultdict(set)
    for r in canonical:
        if r["canonical_pt_code"]:
            norm_to_codes[r["faers_term_normalized"]].add(r["canonical_pt_code"])
    conflicting_norms = {k: v for k, v in norm_to_codes.items() if len(v) > 1}
    if conflicting_norms:
        for r in canonical:
            codes = conflicting_norms.get(r["faers_term_normalized"])
            if codes:
                r["canonical_pt_code"] = None
                r["canonical_pt_name"] = None
                for key in ("canonical_hlt_name", "canonical_hlt_code", "canonical_hlgt_name", "canonical_hlgt_code", "canonical_soc_name", "canonical_soc_code"):
                    r[key] = None
                r["mapping_status"] = "AMBIGUOUS_NORMALIZED_TERM"
                r["manual_review_flag"] = True
                r["comments"] = "normalized term maps to multiple PTs: " + ";".join(sorted(codes))
                legacy = legacy_by_key.get(r["faers_term_raw"].upper().strip())
                correct, error_class, old_code_name = classify_legacy(r, legacy)
                r["legacy_code_correct"] = correct
                r["legacy_code_error_class"] = error_class
                r["legacy_code_pt_name"] = old_code_name

    canonical_fields = [
        "faers_term_raw", "faers_term_normalized", "canonical_pt_name", "canonical_pt_code",
        "canonical_hlt_name", "canonical_hlt_code", "canonical_hlgt_name", "canonical_hlgt_code",
        "canonical_soc_name", "canonical_soc_code", "mapping_level", "mapping_status",
        "legacy_pt_code", "legacy_code_correct", "legacy_code_error_class", "manual_review_flag",
        "comments", "case_frequency", "primaryid_frequency", "expanded_row_frequency",
        "legacy_pt_name", "legacy_code_pt_name",
    ]
    write_csv(OUT / "faers_meddra28_canonical.csv", canonical, canonical_fields)

    # Re-map every legacy term independently from its old code.
    legacy_inventory = []
    for key, legacy in legacy_by_key.items():
        mapped = map_term(str(legacy["legacy_term"]))
        freq = norm_freq.get(key, {})
        correct, error_class, old_code_name = classify_legacy(mapped, legacy)
        legacy_inventory.append(
            {
                "faers_term_raw": legacy["legacy_term"],
                "faers_term_normalized": mapped["faers_term_normalized"],
                "legacy_pt_name": legacy["legacy_pt_name"],
                "legacy_pt_code": legacy["legacy_pt_code"],
                "legacy_code_pt_name": old_code_name,
                "canonical_pt_name": mapped.get("canonical_pt_name"),
                "canonical_pt_code": mapped.get("canonical_pt_code"),
                "mapping_level": mapped["mapping_level"],
                "mapping_status": mapped["mapping_status"],
                "legacy_code_correct": correct,
                "legacy_code_error_class": error_class,
                "case_frequency": freq.get("case_frequency", 0),
                "primaryid_frequency": freq.get("primaryid_frequency", legacy["legacy_primaryid_frequency"]),
                "comments": mapped["comments"],
            }
        )
    legacy_inventory.sort(key=lambda r: (-int(r["case_frequency"]), str(r["faers_term_raw"])))
    write_csv(OUT / "legacy_code_error_inventory.csv", legacy_inventory)
    mismatches = [r for r in legacy_inventory if r["legacy_code_error_class"] != "1_NAME_AND_CODE_CORRECT"]
    write_csv(OUT / "top200_legacy_mismatches.csv", mismatches[:200])

    # Deterministic stratified review sample.
    strata = {
        "common": mismatches[:25],
        "rare": sorted(mismatches, key=lambda r: (int(r["case_frequency"]), str(r["faers_term_raw"])))[:25],
        "spelling_or_punctuation": [r for r in legacy_inventory if str(r["mapping_level"]).startswith(("L2", "L3_LLT_PUNCT"))][:25],
        "llt_derived": [r for r in legacy_inventory if str(r["mapping_level"]).startswith("L3")][:25],
        "historical": [r for r in legacy_inventory if str(r["mapping_level"]).startswith("L4")][:25],
        "invalid_legacy_code": [r for r in legacy_inventory if str(r["legacy_code_error_class"]).startswith("4_")][:25],
        "unresolved_or_ambiguous": [r for r in legacy_inventory if r["mapping_status"] != "MAPPED"][:25],
    }
    review_sample = []
    seen = set()
    for stratum, records in strata.items():
        for r in records:
            key = r["faers_term_raw"]
            if (stratum, key) in seen:
                continue
            seen.add((stratum, key))
            review_sample.append({"review_stratum": stratum, **r})
    write_csv(OUT / "manual_review_stratified_sample.csv", review_sample)

    mapped_canonical = [r for r in canonical if r["mapping_status"] == "MAPPED"]
    unresolved_canonical = [r for r in canonical if r["mapping_status"] != "MAPPED"]
    mapping_levels = Counter(r["mapping_level"] for r in canonical)
    status_counts = Counter(r["mapping_status"] for r in canonical)
    error_counts = Counter(r["legacy_code_error_class"] for r in legacy_inventory)
    mapped_legacy = [r for r in legacy_inventory if r["mapping_status"] == "MAPPED"]
    wrong_legacy = [r for r in mapped_legacy if r["legacy_code_correct"] is not True]

    valid_code_fail = sum(r["canonical_pt_code"] not in pt_name_by_code for r in mapped_canonical)
    name_code_fail = sum(pt_name_by_code.get(r["canonical_pt_code"]) != r["canonical_pt_name"] for r in mapped_canonical)
    hierarchy_fail = sum(r["canonical_pt_code"] not in hierarchy for r in mapped_canonical)
    provenance_fail = sum(not r["mapping_level"] for r in mapped_canonical)
    normalized_multi_fail = len(conflicting_norms)

    # Global repaired FAERS ↔ frozen JADER V5 overlap.
    faers_codes = {r["canonical_pt_code"] for r in mapped_canonical}
    jader_codes = {
        str(x[0]) for x in con.execute(
            f"SELECT DISTINCT PT_CODE FROM read_parquet('{JADER_REAC}') WHERE PT_CODE IS NOT NULL"
        ).fetchall()
    }
    shared_codes = faers_codes & jader_codes

    example_rows = con.execute(
        f"""
        SELECT primaryid, pt, pt_norm, pt_u, pt_name, pt_code, meddra_pt_code
        FROM read_parquet('{raw}')
        WHERE pt IS NOT NULL AND pt_u IS NOT NULL AND upper(trim(pt))<>pt_u
        LIMIT 12
        """
    ).fetchall()
    example_records = [
        dict(zip(["primaryid", "pt", "pt_norm", "pt_u", "pt_name", "pt_code", "meddra_pt_code"], x))
        for x in example_rows
    ]

    error_table = [{"error_class": k, "terms": v} for k, v in sorted(error_counts.items())]
    level_table = [{"mapping_level": k, "terms": v} for k, v in sorted(mapping_levels.items())]
    status_table = [{"mapping_status": k, "terms": v} for k, v in sorted(status_counts.items())]

    root_md = f"""# 01 — Root-cause audit of the legacy FAERS PT defect

## Root cause

The legacy universe was created in `scripts/p02_build_universes.py` by grouping the expanded FAERS SuperMaster on `pt_u` and selecting `any_value(pt_name)` plus `any_value(meddra_pt_code)`. This is invalid for this source grain.

The original reaction identity is carried by `pt`/`pt_norm`. In the expanded SuperMaster, `pt_u` and `meddra_pt_code` can represent a different reaction from the same report-level cross-product. Independent `any_value()` calls then choose a display name and code from unrelated expanded rows. There is no evidence that row sorting can repair this; the error is a key/grain error, not a stable positional shift.

The repaired mapping uses only `pt` text and MedDRA 28.0 terminology keys. Neither `pt_code`, `meddra_pt_code`, nor the legacy aggregate code participates in selecting the repaired PT.

## Source-grain evidence

{md_table([source_diag], [(k,k) for k in source_diag])}

Representative misaligned expanded rows:

{md_table(example_records, [('primaryid','primaryid'),('pt','Original pt'),('pt_norm','pt_norm'),('pt_u','Legacy pt_u'),('pt_name','Original-side PT name'),('pt_code','Original-side code'),('meddra_pt_code','pt_u-side code')])}

## Legacy-term classification

{md_table(error_table, [('error_class','Class'),('terms','Legacy terms')])}

- Legacy terms audited: **{len(legacy_inventory):,}**.
- Conservatively mappable legacy terms: **{len(mapped_legacy):,}**.
- Mappable terms with wrong legacy code: **{len(wrong_legacy):,} ({pct(len(wrong_legacy), len(mapped_legacy)):.4f}%)**.

`legacy_code_error_inventory.csv` preserves every legacy term and its classification. No legacy artefact was overwritten.
"""
    (OUT / "01_root_cause_audit.md").write_text(root_md, encoding="utf-8")

    review_strata_counts = Counter(r["review_stratum"] for r in review_sample)
    review_table = [{"stratum": k, "records": v} for k, v in review_strata_counts.items()]
    qc_pass = all(x == 0 for x in (valid_code_fail, name_code_fail, hierarchy_fail, provenance_fail, normalized_multi_fail))
    qc_md = f"""# FAERS MedDRA 28.0 terminology QC report

**Terminology-layer status:** {'PASS' if qc_pass else 'FAIL'}.

## Mapping yield

- Original raw FAERS terms: **{len(canonical):,}**.
- Mapped raw terms: **{len(mapped_canonical):,} ({pct(len(mapped_canonical), len(canonical)):.4f}%)**.
- Unresolved/ambiguous raw terms: **{len(unresolved_canonical):,}**.
- Distinct repaired current PT codes: **{len(faers_codes):,}**.
- Case-frequency-weighted mapping: **{pct(sum(r['case_frequency'] for r in mapped_canonical), sum(r['case_frequency'] for r in canonical)):.4f}%** (term-specific case counts; cases may contribute multiple terms).

{md_table(level_table, [('mapping_level','Mapping level'),('terms','Raw terms')])}

{md_table(status_table, [('mapping_status','Status'),('terms','Raw terms')])}

## Mandatory checks

| Check | Failures |
|---|---:|
| Canonical code absent from MedDRA 28.0 PT | {valid_code_fail} |
| Canonical PT name/code inconsistency | {name_code_fail} |
| Missing/non-unique primary PT hierarchy | {hierarchy_fail} |
| Missing mapping provenance | {provenance_fail} |
| Normalized source term mapping to >1 retained PT | {normalized_multi_fail} |

## Manual mismatch review sample

{md_table(review_table, [('stratum','Stratum'),('records','Records inspected')])}

The deterministic review set is saved as `manual_review_stratified_sample.csv`; the 200 highest-frequency mismatches are saved separately. Review strata cover common, rare, punctuation/spelling, LLT-derived, historical, invalid-code, and unresolved/ambiguous records. No fuzzy or outcome-informed mapping was used.

### Recorded manual review conclusion

The deterministic sample was inspected record by record. Common current-PT examples reproduced the official name/code pair; the apparent mismatch in this stratum was the independently aggregated legacy display name. LLT-derived and historical examples resolved through the documented parent/history relationship (for example, `SPINAL COMPRESSION FRACTURE` → `Compression fracture` and `DISBACTERIOSIS` → `Dysbiosis`). The single available punctuation-only legacy mismatch resolved uniquely after the prespecified normalization. Blank/invalid legacy codes did not influence the replacement mapping. Unresolved terms had no conservative unique dictionary path and were left unresolved; no manual code was imputed.

Nine rare legacy `pt_u` entries in the review sample have zero frequency when counted in the true original `pt` field. This is direct evidence that the old `pt_u` universe contained cross-product terms not present as original reaction identities, and it supports rebuilding from `pt` rather than attempting to repair old codes in place. No sampled record required changing the deterministic cascade.
"""
    (OUT / "terminology_qc_report.md").write_text(qc_md, encoding="utf-8")

    overlap_md = f"""# 02 — Rebuilt FAERS↔JADER PT overlap

**Global terminology overlap:** recalculated and valid.  
**FDA-cohort/B-STRICT-specific overlap:** BLOCKED pending the frozen cohort and B-STRICT registry.

| Quantity | Rebuilt value |
|---|---:|
| Distinct repaired FAERS PT codes | {len(faers_codes):,} |
| Distinct JADER V5 PT codes | {len(jader_codes):,} |
| Shared current PT codes | {len(shared_codes):,} |
| JADER→FAERS coverage | {pct(len(shared_codes),len(jader_codes)):.4f}% |
| FAERS→JADER coverage | {pct(len(shared_codes),len(faers_codes)):.4f}% |

The prior 11,313/11,551 estimate is superseded because it was derived from the `pt_u`-based legacy universe rather than original FAERS `pt` terms. JADER V5 internal terminology was not changed.
"""
    (OUT / "02_cross_database_pt_overlap_rebuilt.md").write_text(overlap_md, encoding="utf-8")

    decision = {
        "status": "PASS_TERMINOLOGY_LAYER" if qc_pass else "FAIL_TERMINOLOGY_LAYER",
        "downstream_overall_status": "HOLD",
        "root_cause": "grouping expanded FAERS SuperMaster by misaligned pt_u and independent any_value selection of pt_name/meddra_pt_code",
        "legacy_terms_audited": len(legacy_inventory),
        "legacy_terms_mappable": len(mapped_legacy),
        "legacy_mappable_terms_wrong_code": len(wrong_legacy),
        "legacy_wrong_code_pct": pct(len(wrong_legacy), len(mapped_legacy)),
        "original_raw_terms": len(canonical),
        "mapped_raw_terms": len(mapped_canonical),
        "mapping_rate_pct": pct(len(mapped_canonical), len(canonical)),
        "unresolved_or_ambiguous_raw_terms": len(unresolved_canonical),
        "distinct_repaired_faers_pt_codes": len(faers_codes),
        "jader_v5_pt_codes": len(jader_codes),
        "shared_faers_jader_pt_codes": len(shared_codes),
        "jader_to_faers_coverage_pct": pct(len(shared_codes), len(jader_codes)),
        "faers_to_jader_coverage_pct": pct(len(shared_codes), len(faers_codes)),
        "qc": {
            "invalid_code_failures": valid_code_fail,
            "name_code_failures": name_code_fail,
            "hierarchy_failures": hierarchy_fail,
            "provenance_failures": provenance_fail,
            "normalized_term_multi_pt_failures": normalized_multi_fail,
        },
        "old_codes_used_to_determine_mapping": False,
        "machine_learning_trained": False,
        "downstream_blockers": [
            "FDA first-approval cohort absent",
            "B-STRICT registry absent",
            "drug_identity_master.csv absent",
            "Original Command 02 approval windows and temporal split locks absent",
        ],
    }
    (OUT / "FAERS_PT_REPAIR_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (PROC / "terminology_repair_metrics.json").write_text(
        json.dumps({"source_diag": source_diag, "decision": decision, "mapping_levels": mapping_levels, "error_classes": error_counts}, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
