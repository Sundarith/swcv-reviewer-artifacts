"""Validate saved retrieval, reconstruct votes, and recompute strict accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
from .voting import predict

def records(path):
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {value["question_id"]: value for value in values}
    if len(indexed) != len(values):
        raise ValueError(f"Duplicate question IDs: {path}")
    return indexed

def reproduce(root, run):
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")
    directory = root / "artifacts" / run
    retrieval = records(directory / "retrieval.jsonl")
    predictions = records(directory / "predictions.jsonl")
    gold = records(root / "benchmarks" / "ctibench_rcm1000_gold.jsonl")
    selection = records(root / "benchmarks" / "ctibench_rcm1000_ids.jsonl")
    if not (retrieval.keys() == predictions.keys() == gold.keys() == selection.keys()):
        raise ValueError("Question coverage mismatch")
    excluded = set(json.loads((directory / "config.json").read_text())["excluded_reference_cve_ids"])
    for key, entry in retrieval.items():
        if len(entry["neighbors"]) != 20:
            raise ValueError("Expected exactly twenty neighbors")
        if any(n["cve_id"] in excluded for n in entry["neighbors"]):
            raise ValueError("Excluded reference found")
        if predict(entry["neighbors"]) != predictions[key]["prediction"]:
            raise ValueError(f"Vote mismatch: {key}")
    groups = {"full1000": set(selection), "original335": set(records(root / "benchmarks" / "ctibench_original335_ids.jsonl"))}
    groups["remaining665"] = groups["full1000"] - groups["original335"]
    result = {}
    for name, ids in groups.items():
        correct = sum(predictions[key]["prediction"] == gold[key]["gold"] for key in ids)
        result[name] = {"correct": correct, "n": len(ids), "accuracy_percent": 100 * correct / len(ids)}
    if result != json.loads((directory / "scores.json").read_text()):
        raise ValueError("Recomputed scores differ from saved scores")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", choices=["swcv_exact_baseline", "swcv_union_filtered"], required=True)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.root, args.run), indent=2))

if __name__ == "__main__":
    main()
