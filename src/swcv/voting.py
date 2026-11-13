"""Original sum-of-cosines vote, preserving lexical CWE tie-breaking."""
from collections import Counter
import math

def predict(neighbors):
    weights = Counter()
    for neighbor in neighbors:
        similarity = float(neighbor["similarity"])
        if not math.isfinite(similarity):
            raise ValueError("Similarity must be finite")
        for label in neighbor["cwe_ids"]:
            weights[label] += similarity
    if not weights:
        raise ValueError("No labeled neighbors")
    return sorted(weights, key=lambda label: (-weights[label], label))[0]
