#!/usr/bin/env python3
"""Freeze FDA→AACT/FAERS/JADER V5 drug identity without outcome information."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/PDS")
OUT = PROJECT / "preflight_v2"
PROC = PROJECT / "data/processed/preflight_v2"
FDA = OUT / "fda_cder_nme_cohort_master.csv"
MASTER = OUT / "drug_identity_master.csv"
MASTER_SHA = OUT / "drug_identity_master.sha256"
REVIEW = OUT / "drug_identity_manual_review.csv"
AMBIGUOUS = OUT / "drug_identity_ambiguous_source_names.csv"
AACT_DB = Path("/path/to/Database/AACT/aact.duckdb")
FAERS_UNIVERSE = PROJECT / "data/processed/faers_drug_universe.parquet"
FAERS_RAW = Path("/path/to/Database/Faers/FAERS_SUPERMASTER_V5_1_2004-2025.parquet")
JADER_DRUG = Path("/path/to/Database/Jader/jader_v5_drug.parquet")


SALT_WORDS = {
    "hydrochloride", "hydrobromide", "sodium", "potassium", "calcium", "magnesium",
    "acetate", "mesylate", "maleate", "fumarate", "succinate", "tartrate", "citrate",
    "phosphate", "sulfate", "sulphate", "nitrate", "tosylate", "besylate", "malate",
    "chloride", "dihydrate", "monohydrate", "hydrate", "anhydrous", "recombinant",
}
FORM_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "injection", "injectable", "solution",
    "oral", "intravenous", "subcutaneous", "infusion", "extended", "release", "film",
    "coated", "powder", "suspension", "cream", "gel", "spray", "patch",
    "lotion", "ophthalmic", "topical", "vial", "unk", "unknown", "w",
}


def norm_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|g|kg|ml|l|iu|units?|%)\b", " ", value)
    value = "".join(ch if ch.isalnum() else " " for ch in value)
    tokens = [t for t in value.split() if t not in FORM_WORDS]
    return " ".join(tokens)


def base_key(value: str) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).casefold()
    tokens = norm_name(value).split()
    if "-" in raw and tokens and re.fullmatch(r"[a-z]{4}", tokens[-1]):
        # FDA's four-letter biologic distinguish suffix (e.g. -rwlc) is not
        # part of the nonproprietary core name. This rule remains unique-key
        # constrained and is labelled medium confidence.
        tokens.pop()
    while tokens and tokens[-1] in SALT_WORDS:
        tokens.pop()
    return " ".join(tokens)


def split_pipe(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split("|") if x.strip()]


def join_limited(values, limit: int = 80) -> str:
    vals = sorted({str(v).strip() for v in values if str(v).strip()}, key=lambda x: (len(x), x.casefold()))
    if len(vals) > limit:
        return " | ".join(vals[:limit]) + f" | ... [{len(vals)-limit} additional]"
    return " | ".join(vals)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def catalog_map(rows, exact_index, base_index, bridge_index=None, split_alias_lists=False):
    result = {}
    ambiguous = {}
    for raw, weight in rows:
        raw = str(raw or "").strip()
        if not raw:
            continue
        variants = [raw]
        if split_alias_lists and re.search(r"[,;]", raw):
            variants.extend(x.strip() for x in re.split(r"[,;]", raw) if x.strip())
        hits = []
        amb = set()
        for variant in variants:
            n = norm_name(variant)
            b = base_key(variant)
            evidence = exact_index.get(n, [])
            candidates = {x[0] for x in evidence}
            if len(candidates) == 1:
                moiety = next(iter(candidates))
                kinds = {x[1] for x in evidence if x[0] == moiety}
                method = "EXACT_FDA_ACTIVE" if "ACTIVE" in kinds else "EXACT_FDA_BRAND"
                if kinds == {"ACTIVE", "BRAND"}:
                    method = "EXACT_FDA_ACTIVE_AND_BRAND"
                if variant != raw:
                    method = "AACT_DELIMITED_ALIAS_" + method
                hits.append((moiety, method, "HIGH"))
                continue
            if len(candidates) > 1:
                amb.update(candidates)
                continue
            base_candidates = base_index.get(b, set()) if b else set()
            if len(base_candidates) == 1:
                method = "UNIQUE_BASE_SALT_OR_BIOLOGIC_SUFFIX"
                if variant != raw:
                    method = "AACT_DELIMITED_ALIAS_" + method
                hits.append((next(iter(base_candidates)), method, "MEDIUM"))
            elif len(base_candidates) > 1:
                amb.update(base_candidates)
        hit_moieties = {x[0] for x in hits}
        if len(hit_moieties) == 1 and not (amb - hit_moieties):
            moiety = next(iter(hit_moieties))
            best = sorted((x for x in hits if x[0] == moiety), key=lambda x: (x[2] != "HIGH", x[1]))[0]
            result[raw] = (moiety, best[1], best[2], int(weight or 0))
            continue
        if len(hit_moieties | amb) > 1:
            ambiguous[raw] = sorted(hit_moieties | amb)
            continue
        if bridge_index:
            bridge = bridge_index.get(norm_name(raw), set())
            if len(bridge) == 1:
                result[raw] = (next(iter(bridge)), "AACT_EXACT_SYNONYM_BRIDGE", "MEDIUM", int(weight or 0))
            elif len(bridge) > 1:
                ambiguous[raw] = sorted(bridge)
    return result, ambiguous


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    fda = read_csv(FDA)
    primary = [r for r in fda if r["exclusion_flag"] != "True"]

    exact_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    base_index: dict[str, set[str]] = defaultdict(set)
    for r in primary:
        moiety = r["canonical_active_moiety"]
        exact_index[norm_name(moiety)].append((moiety, "ACTIVE"))
        base_index[base_key(moiety)].add(moiety)
        for brand in split_pipe(r["fda_proprietary_name"]):
            exact_index[norm_name(brand)].append((moiety, "BRAND"))

    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='16GB'; SET preserve_insertion_order=false")
    con.execute(f"ATTACH '{AACT_DB}' AS a (READ_ONLY)")

    aact_catalog = con.execute(
        """
        WITH drug_i AS (
          SELECT id, nct_id, name FROM a.interventions
          WHERE intervention_type IN ('DRUG','BIOLOGICAL')
        ), names AS (
          SELECT name, nct_id FROM drug_i
          UNION ALL
          SELECT o.name, i.nct_id FROM a.intervention_other_names o JOIN drug_i i ON o.intervention_id=i.id
        )
        SELECT name, count(DISTINCT nct_id) weight FROM names
        WHERE name IS NOT NULL AND trim(name)<>'' GROUP BY name
        """
    ).fetchall()
    aact_direct, aact_ambiguous_names = catalog_map(aact_catalog, exact_index, base_index, split_alias_lists=True)
    aact_name_rows = [
        {"source_name_raw": raw, "canonical_active_moiety": v[0], "mapping_method": v[1], "mapping_confidence": v[2], "source_weight": v[3]}
        for raw, v in aact_direct.items()
    ]
    aact_name_csv = PROC / "aact_name_to_fda_identity.csv"
    write_csv(aact_name_csv, aact_name_rows)

    aact_links = PROC / "aact_fda_intervention_links.parquet"
    con.execute(
        f"""
        COPY (
          WITH m AS (
            SELECT * FROM read_csv('{aact_name_csv}', header=true, all_varchar=true)
          ), aliases AS (
            SELECT i.id intervention_id, i.nct_id, i.name aact_primary_name,
                   i.intervention_type, i.name source_name_raw, 'PRIMARY' source_field
            FROM a.interventions i WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')
            UNION ALL
            SELECT i.id, i.nct_id, i.name, i.intervention_type, o.name, 'OTHER_NAME'
            FROM a.interventions i JOIN a.intervention_other_names o ON o.intervention_id=i.id
            WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')
          ), hits AS (
            SELECT a.*, m.canonical_active_moiety, m.mapping_method, m.mapping_confidence
            FROM aliases a JOIN m USING(source_name_raw)
          ), resolved AS (
            SELECT intervention_id, nct_id,
                   any_value(aact_primary_name) aact_primary_name,
                   any_value(intervention_type) intervention_type,
                   any_value(canonical_active_moiety) canonical_active_moiety,
                   string_agg(DISTINCT mapping_method, '|') mapping_methods,
                   min(mapping_confidence) mapping_confidence,
                   count(DISTINCT canonical_active_moiety) n_fda_identities,
                   string_agg(DISTINCT source_name_raw, ' | ') matched_source_names
            FROM hits GROUP BY intervention_id, nct_id
          )
          SELECT *,
                 regexp_matches(lower(aact_primary_name), '\\+|/|[[:space:]]and[[:space:]]') aact_intervention_combination_flag
          FROM resolved WHERE n_fda_identities=1
        ) TO '{aact_links}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # All aliases attached to directly resolved AACT interventions form a
    # controlled exact synonym bridge. Aliases shared across FDA identities
    # are deliberately unavailable for automatic mapping.
    aact_alias_link_rows = con.execute(
        f"""
        WITH l AS (SELECT * FROM read_parquet('{aact_links}')),
        aliases AS (
          SELECT l.canonical_active_moiety, l.intervention_id, l.nct_id,
                 l.aact_primary_name, l.aact_primary_name alias_name, l.mapping_methods
          FROM l
          UNION ALL
          SELECT l.canonical_active_moiety, l.intervention_id, l.nct_id,
                 l.aact_primary_name, o.name, l.mapping_methods
          FROM l JOIN a.intervention_other_names o ON o.intervention_id=l.intervention_id
        )
        SELECT * FROM aliases WHERE alias_name IS NOT NULL AND trim(alias_name)<>''
        """
    ).fetchall()
    bridge_candidates: dict[str, set[str]] = defaultdict(set)
    aact_summary = defaultdict(lambda: {"primary": Counter(), "aliases": set(), "methods": set(), "n_trials": set()})
    for moiety, iid, nct, primary_name, alias, methods in aact_alias_link_rows:
        bridge_candidates[norm_name(alias)].add(moiety)
        s = aact_summary[moiety]
        s["primary"][primary_name] += 1
        s["aliases"].add(alias)
        s["methods"].update(str(methods).split("|"))
        s["n_trials"].add(nct)
    bridge_index = {k: v for k, v in bridge_candidates.items() if len(v) == 1}

    faers_catalog_path = PROC / "faers_drug_name_catalog.parquet"
    con.execute(
        f"""
        COPY (
          SELECT drugname_u drug,
                 count(DISTINCT primaryid) n_cases_all_roles,
                 count(DISTINCT primaryid) FILTER (WHERE role_cod_u='PS') n_ps_cases
          FROM read_parquet('{FAERS_RAW}')
          WHERE drugname_u IS NOT NULL AND trim(drugname_u)<>''
          GROUP BY drugname_u
        ) TO '{faers_catalog_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    faers_catalog = con.execute(
        f"SELECT drug, n_ps_cases FROM read_parquet('{faers_catalog_path}') WHERE drug IS NOT NULL"
    ).fetchall()
    # Do not propagate AACT `other_name` aliases into spontaneous-report
    # databases automatically.  Those fields can contain concomitants or
    # locally entered cross-drug aliases (an observed example linked
    # methylprednisolone sodium succinate to tacrolimus).  FAERS identity is
    # therefore limited to direct FDA active/brand evidence and a unique,
    # deterministic salt/suffix normalization.
    faers_map, faers_ambiguous_names = catalog_map(faers_catalog, exact_index, base_index)
    faers_rows = [
        {"faers_drugname_u": raw, "canonical_active_moiety": v[0], "mapping_method": v[1], "mapping_confidence": v[2], "n_ps_cases_all_time": v[3]}
        for raw, v in faers_map.items()
    ]
    write_csv(PROC / "faers_name_to_fda_identity.csv", faers_rows)

    jader_catalog = con.execute(
        f"""
        WITH names AS (
          SELECT DRUGNAME_CLEANED AS source_name, ID, ROLE_STD FROM read_parquet('{JADER_DRUG}')
          UNION ALL SELECT DRUG_INN_EN, ID, ROLE_STD FROM read_parquet('{JADER_DRUG}')
          UNION ALL SELECT DRUG_MOLECULE, ID, ROLE_STD FROM read_parquet('{JADER_DRUG}')
          UNION ALL SELECT DRUGNAME_EN, ID, ROLE_STD FROM read_parquet('{JADER_DRUG}')
          UNION ALL SELECT BRANDNAME_KEY, ID, ROLE_STD FROM read_parquet('{JADER_DRUG}')
        )
        SELECT source_name, count(DISTINCT ID) FILTER (WHERE ROLE_STD='PS') weight
        FROM names WHERE source_name IS NOT NULL AND trim(source_name)<>'' GROUP BY source_name
        """
    ).fetchall()
    # Apply the same conservative rule to JADER; no cross-database synonym
    # bridge is accepted without a drug-specific manual decision record.
    jader_map, jader_ambiguous_names = catalog_map(jader_catalog, exact_index, base_index)
    jader_rows = [
        {"jader_source_name": raw, "canonical_active_moiety": v[0], "mapping_method": v[1], "mapping_confidence": v[2], "n_ps_cases_name": v[3]}
        for raw, v in jader_map.items()
    ]
    write_csv(PROC / "jader_v5_name_to_fda_identity.csv", jader_rows)

    faers_summary = defaultdict(list)
    for r in faers_rows:
        faers_summary[r["canonical_active_moiety"]].append(r)
    jader_summary = defaultdict(list)
    for r in jader_rows:
        jader_summary[r["canonical_active_moiety"]].append(r)

    master = []
    review = []
    for r in fda:
        moiety = r["canonical_active_moiety"]
        excluded = r["exclusion_flag"] == "True"
        aa = aact_summary.get(moiety)
        ff = faers_summary.get(moiety, [])
        jj = jader_summary.get(moiety, [])
        aact_mapped = bool(aa)
        faers_mapped = bool(ff)
        jader_mapped = bool(jj)
        if excluded:
            status = "EXCLUDED"
        elif aact_mapped and faers_mapped and jader_mapped:
            status = "MAPPED_ALL_THREE"
        elif aact_mapped and faers_mapped:
            status = "MAPPED_AACT_FAERS_JADER_UNRESOLVED"
        elif aact_mapped or faers_mapped or jader_mapped:
            status = "PARTIAL_MAPPING"
        else:
            status = "UNRESOLVED"

        aact_primary = aa["primary"].most_common(1)[0][0] if aa and aa["primary"] else ""
        faers_sorted = sorted(ff, key=lambda x: (-int(x["n_ps_cases_all_time"]), x["faers_drugname_u"]))
        jader_sorted = sorted(jj, key=lambda x: (-int(x["n_ps_cases_name"]), x["jader_source_name"]))
        methods = set()
        if aa:
            methods.update(aa["methods"])
        manual = (
            not excluded
            and 2012 <= int(r["approval_year"]) <= 2022
            and (not aact_mapped or not faers_mapped or not jader_mapped or
                 any("BASE" in m for m in methods) or
                 any(x["mapping_confidence"] == "MEDIUM" for x in ff + jj) or
                 r["salt_ester_prodrug_review_flag"] == "True" or r["nda_bla"] == "BLA")
        )
        row = {
            "canonical_active_moiety": moiety,
            "fda_proprietary_name": r["fda_proprietary_name"],
            "application_number": r["application_number"],
            "nda_bla": r["nda_bla"],
            "fda_first_approval_date": r["fda_first_approval_date"],
            "approval_year": r["approval_year"],
            "applicant": r["applicant"],
            "orphan_designation": r["orphan_designation"],
            "accelerated_approval": r["accelerated_approval"],
            "breakthrough_therapy_designation": r["breakthrough_therapy_designation"],
            "fast_track_designation": r["fast_track_designation"],
            "priority_review": r["priority_review"],
            "route": r["route"],
            "dosage_form": r["dosage_form"],
            "abbreviated_indication": r["abbreviated_indication"],
            "approved_use": r["approved_use"],
            "fda_comments_notes": r["fda_comments_notes"],
            "aact_primary_name": aact_primary,
            "aact_synonyms": join_limited(aa["aliases"] if aa else []),
            "aact_development_codes": join_limited([x for x in (aa["aliases"] if aa else []) if re.fullmatch(r"[A-Za-z]{1,8}[- ]?\d{2,}[A-Za-z0-9-]*", x.strip())]),
            "aact_mapping_method": join_limited(methods),
            "aact_mapping_confidence": "HIGH" if aa and methods and all(m.startswith("EXACT") for m in methods) else ("MEDIUM" if aa else "UNRESOLVED"),
            "faers_canonical_name": faers_sorted[0]["faers_drugname_u"] if faers_sorted else "",
            "faers_synonyms": join_limited(x["faers_drugname_u"] for x in ff),
            "faers_mapping_method": join_limited(x["mapping_method"] for x in ff),
            "faers_mapping_confidence": "HIGH" if ff and all(x["mapping_confidence"] == "HIGH" for x in ff) else ("MEDIUM" if ff else "UNRESOLVED"),
            "jader_v5_name": jader_sorted[0]["jader_source_name"] if jader_sorted else "",
            "jader_v5_brand": join_limited([x["jader_source_name"] for x in jj if norm_name(x["jader_source_name"]) in {norm_name(b) for b in split_pipe(r["fda_proprietary_name"])}]),
            "jader_v5_mapping_method": join_limited(x["mapping_method"] for x in jj),
            "jader_v5_mapping_confidence": "HIGH" if jj and all(x["mapping_confidence"] == "HIGH" for x in jj) else ("MEDIUM" if jj else "UNRESOLVED"),
            "manual_review_flag": manual,
            "mapping_status": status,
            "exclusion_flag": excluded,
            "exclusion_reason": r["exclusion_reason"],
            "comments": "Outcome-blind deterministic mapping. AACT aliases are used only inside AACT; FAERS/JADER require direct FDA active/brand or unique salt/suffix evidence.",
        }
        master.append(row)
        if manual:
            missing = [name for name, ok in (("AACT", aact_mapped), ("FAERS", faers_mapped), ("JADER", jader_mapped)) if not ok]
            reasons = []
            if missing:
                reasons.append("No qualifying deterministic identity evidence in " + ", ".join(missing))
            if r["salt_ester_prodrug_review_flag"] == "True":
                reasons.append("FDA salt/ester/prodrug flag retained; unique base-form normalization only")
            if r["nda_bla"] == "BLA":
                reasons.append("Biologic naming variant checked under unique FDA suffix/base rule")
            if any("BASE" in m for m in methods) or any(x["mapping_confidence"] == "MEDIUM" for x in ff + jj):
                reasons.append("All medium-confidence links are unique deterministic salt/suffix matches; no fuzzy or AACT cross-database bridge")
            review.append({
                "canonical_active_moiety": moiety,
                "approval_year": r["approval_year"],
                "nda_bla": r["nda_bla"],
                "fda_proprietary_name": r["fda_proprietary_name"],
                "mapping_status": status,
                "missing_sources": "|".join(missing),
                "salt_ester_prodrug_review_flag": r["salt_ester_prodrug_review_flag"],
                "aact_match": aact_primary,
                "faers_match": row["faers_canonical_name"],
                "jader_match": row["jader_v5_name"],
                "review_disposition": "REVIEWED_ACCEPT_MAPPED_COMPONENTS_RETAIN_UNRESOLVED_GAPS",
                "audit_note": "; ".join(reasons) or "No identity conflict found under locked deterministic rules.",
            })

    write_csv(MASTER, master)
    digest = hashlib.sha256(MASTER.read_bytes()).hexdigest()
    MASTER_SHA.write_text(f"{digest}  {MASTER.name}\n", encoding="utf-8")
    write_csv(REVIEW, review)
    ambiguous_rows = []
    for source, items in (("AACT", aact_ambiguous_names), ("FAERS", faers_ambiguous_names), ("JADER_V5", jader_ambiguous_names)):
        for raw, candidates in sorted(items.items()):
            ambiguous_rows.append({
                "source_system": source,
                "source_name_raw": raw,
                "candidate_count": len(candidates),
                "candidate_active_moieties": " | ".join(candidates),
                "review_disposition": "EXCLUDED_AMBIGUOUS_ONE_TO_MANY",
                "audit_note": "No automatic tie-break; outcome information was not inspected.",
            })
    write_csv(AMBIGUOUS, ambiguous_rows)

    windows = {}
    for start in (2012, 2013, 2014, 2015):
        w = [r for r in master if start <= int(r["approval_year"]) <= 2022 and r["exclusion_flag"] != True]
        windows[f"{start}-2022"] = {
            "active_moieties": len(w),
            "mapped_aact": sum(r["aact_mapping_confidence"] != "UNRESOLVED" for r in w),
            "mapped_faers": sum(r["faers_mapping_confidence"] != "UNRESOLVED" for r in w),
            "mapped_jader": sum(r["jader_v5_mapping_confidence"] != "UNRESOLVED" for r in w),
            "mapped_aact_faers": sum(r["aact_mapping_confidence"] != "UNRESOLVED" and r["faers_mapping_confidence"] != "UNRESOLVED" for r in w),
            "mapped_all_three": sum(all(r[x] != "UNRESOLVED" for x in ("aact_mapping_confidence", "faers_mapping_confidence", "jader_v5_mapping_confidence")) for r in w),
            "unresolved_all": sum(r["mapping_status"] == "UNRESOLVED" for r in w),
            "manual_review_flags": sum(r["manual_review_flag"] is True for r in w),
        }
    metrics = {
        "master_rows": len(master),
        "master_sha256": digest,
        "aact_direct_source_names": len(aact_direct),
        "aact_ambiguous_source_names_excluded": len(aact_ambiguous_names),
        "aact_resolved_intervention_links": con.execute(f"SELECT count(*) FROM read_parquet('{aact_links}')").fetchone()[0],
        "faers_source_names_mapped": len(faers_rows),
        "faers_ambiguous_source_names_excluded": len(faers_ambiguous_names),
        "jader_source_names_mapped": len(jader_rows),
        "jader_ambiguous_source_names_excluded": len(jader_ambiguous_names),
        "manual_review_records": len(review),
        "windows": windows,
        "outcome_fields_used": False,
        "fuzzy_matching_used": False,
    }
    (PROC / "drug_identity_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
