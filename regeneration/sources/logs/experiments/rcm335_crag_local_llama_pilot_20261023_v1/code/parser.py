"""Post-run exact single-ID scoring, separate from frozen final-line compliance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

ID_RE = re.compile(r"\bCWE-(\d+)\b", re.I)


def final_answer_id(answer, finish_reason):
    if finish_reason != "stop":
        return None
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    elif "<think>" in answer:
        return None
    ids = {f"CWE-{int(x)}" for x in ID_RE.findall(answer)}
    # No gold-dependent selection or credit for listing several possibilities.
    return next(iter(ids)) if len(ids) == 1 else None


def selected_answer_id(answer, finish_reason):
    if finish_reason != "stop":
        return None
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    elif "<think>" in answer:
        return None
    lines = [re.sub(r"[*`]", "", line).strip().lstrip("# ")
             for line in answer.splitlines() if line.strip()]
    selections = set()
    label = re.compile(r"^(?:final\s+answer|answer|selected\s+cwe(?:\s+id)?|cwe\s+(?:id|mapping))\s*[:=]\s*(.+)$", re.I)
    for line in lines:
        match = label.match(line)
        if match:
            ids = {f"CWE-{int(n)}" for n in ID_RE.findall(match[1])}
            if len(ids) != 1:
                return None
            selections.update(ids)
    if lines:
        tail = re.fullmatch(r"CWE-(\d+)[.]?", lines[-1], re.I)
        if tail:
            selections.add(f"CWE-{int(tail[1])}")
        if re.match(r"^CWE-\d+\b", lines[0], re.I):
            ids = {f"CWE-{int(n)}" for n in ID_RE.findall(lines[0])}
            if len(ids) != 1:
                return None
            selections.update(ids)
    if selections:
        return next(iter(selections)) if len(selections) == 1 else None
    return final_answer_id(answer, finish_reason)


def catalog_answer_id(answer, finish_reason, valid_ids):
    selected = selected_answer_id(answer, finish_reason)
    return selected if selected in valid_ids else None


def score(out, root, use_repairs=False, parser="single_id"):
    protocol = json.loads((out / "protocol.json").read_text())
    bench = root / "data/cti-bench/data/cti-rcm.tsv"
    assert hashlib.sha256(bench.read_bytes()).hexdigest() == protocol["sha256"][str(bench)]
    selection = {x["cve_id"]: x for x in json.loads((out / "selection.json").read_text())}
    raw = [json.loads(line) for line in (out / "predictions.jsonl").open()]
    assert len(raw) == 393 and {r["cve_id"] for r in raw} == set(selection)
    if use_repairs:
        replacements = {r["cve_id"]: r for r in map(json.loads, (out / "input_repair_predictions.jsonl").open())}
        raw = [replacements.get(r["cve_id"], r) for r in raw]
    # Extract and save predictions before accessing gold labels.
    parse = selected_answer_id if parser == "explicit" else final_answer_id
    predictions = [{"cve_id": r["cve_id"], "row_index": r["row_index"],
                    "prediction": parse(r.get("answer", ""), r.get("finish_reason")),
                    "finish_reason": r.get("finish_reason"), "error": r.get("error")}
                   for r in raw]
    tag = "repaired_final_answer" if use_repairs else "final_answer"
    if parser == "explicit":
        tag = "repaired_explicit_answer" if use_repairs else "explicit_answer"
    (out / f"{tag}_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    gold_rows = list(csv.DictReader(bench.open(), delimiter="\t"))
    clones = set(json.loads((out / "reference_quality_audit.json").read_text())["exact_description_clones_among_reference_cves"])
    groups = [("all_393", lambda s: True), ("subset_335", lambda s: s["subset_335"]),
              ("no_explicit_cwe_in_prompt_392", lambda s: not s["prompt_has_cwe_id"]),
              ("original_unmapped_97", lambda s: not s["adp_mapped"]),
              ("subset_335_without_reference_description_clones_331", lambda s: s["subset_335"] and s["cve_id"] not in clones)]
    summary = {}
    for name, eligible in groups:
        group = [p for p in predictions if eligible(selection[p["cve_id"]])]
        gold = {p["cve_id"]: final_answer_id(gold_rows[p["row_index"]]["GT"], "stop") for p in group}
        assert all(gold.values())
        correct = sum(p["prediction"] == gold[p["cve_id"]] for p in group)
        summary[name] = {"correct": correct, "n": len(group), "accuracy": correct/len(group),
                         "truncated": sum(p["finish_reason"] == "length" for p in group),
                         "unparseable_or_ambiguous_stopped": sum(p["finish_reason"] == "stop" and p["prediction"] is None for p in group),
                         "errors": sum(bool(p["error"]) for p in group)}
    report = {"status": "Post-run scoring diagnostic; not the originally frozen final-line-only metric.",
              "parser": parser,
              "rule": ("For stop completions, remove tagged reasoning; prioritize consistent explicit answer labels, a final standalone ID, or a first-line CWE heading. If absent, require a single distinct ID. Reject conflicting selections, missing IDs and truncated outputs; normalize padding and grant no hierarchy credit." if parser == "explicit" else "For stop completions, remove tagged reasoning, normalize numeric CWE IDs and require exactly one distinct ID in the final answer. Truncated, missing and ambiguous answers count incorrect. No hierarchy credit."),
              "reason": "The frozen scorer measured output-format compliance; most completed answers identify one CWE but also include explanations or ANSWER prefixes.",
              "generation_calls_added": len(replacements) if use_repairs else 0,
              "input_repairs_used": list(replacements) if use_repairs else [],
              "original_final_line_summary": json.loads((out / "summary.json").read_text()),
              "finish_reasons": dict(Counter(p["finish_reason"] for p in predictions)),
              "results": summary}
    (out / f"{tag}_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--use-repairs", action="store_true")
    parser.add_argument("--parser", choices=["single_id", "explicit"], default="explicit")
    args = parser.parse_args()
    score(args.out.resolve(), args.root.resolve(), args.use_repairs, args.parser)
