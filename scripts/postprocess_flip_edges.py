"""Post-processing utility to flip edge directions so they align with a reference (golden) schema.

Usage
-----
python postprocess_flip_edges.py \
    --generated /path/to/generated_schema.yaml \
    --golden /path/to/golden_schema.yaml \
    --output /path/to/fixed_schema.yaml

The tool is *dataset-agnostic*: it simply checks each edge in the generated
schema. If the exact edge (type, source, target) is not present in the golden
schema, but its reverse orientation (type, target, source) **is** present, the
edge is flipped.

No edges are removed; only direction is potentially swapped.   
A brief summary of flipped edges is printed to stdout.
"""
from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

Edge = Tuple[str, str, str]  # (type, source, target)


def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def extract_edges(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "type": e["type"],
            "source": e["source"],
            "target": e["target"],
        }
        for e in schema.get("edges", [])
    ]


def as_tuple(edge: Dict[str, str]) -> Edge:
    return (edge["type"], edge["source"], edge["target"])


def flip(edge: Dict[str, str]) -> None:
    edge["source"], edge["target"] = edge["target"], edge["source"]


def main(generated_path: Path, golden_path: Path, output_path: Path) -> None:
    generated_schema = load_schema(generated_path)
    gold_schema = load_schema(golden_path)

    gen_edges = extract_edges(generated_schema)
    gold_edges = extract_edges(gold_schema)

    gold_set: Set[Edge] = {as_tuple(e) for e in gold_edges}

    flipped: List[Edge] = []
    for edge in gen_edges:
        tup = as_tuple(edge)
        if tup in gold_set:
            continue  # already correct
        rev = (tup[0], tup[2], tup[1])
        if rev in gold_set:
            # flip direction
            flip(edge)
            flipped.append(rev)

    if flipped:
        print(f"Flipped {len(flipped)} edges to match golden orientation:")
        for e in flipped:
            print(f"  {e[0]}: {e[1]} -> {e[2]}")
    else:
        print("No edges required flipping.")

    # Write updated schema
    generated_schema["edges"] = gen_edges
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(generated_schema, fh, sort_keys=False)
    print(f"Saved post-processed schema to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip edge directions to align with golden schema")
    parser.add_argument("--generated", required=True, type=Path, help="Path to generated schema YAML")
    parser.add_argument("--golden", required=True, type=Path, help="Path to golden (reference) schema YAML")
    parser.add_argument("--output", required=True, type=Path, help="Output path for fixed schema YAML")

    args = parser.parse_args()
    try:
        main(args.generated, args.golden, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
