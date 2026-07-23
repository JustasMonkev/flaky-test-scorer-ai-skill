---
name: flaky-test-scorer
description: >-
  Score and rank flaky tests from CSV or JSON pass/fail run history. Use when a
  user asks to find, prioritize, triage, quarantine, or summarize flaky tests,
  intermittent failures, non-deterministic tests, CI instability, or uploaded
  test-run results that need flakiness analysis.
---

# Flaky Test Scorer

Use this workflow to turn repeated pass/fail test-run history into a ranked flakiness report.  
The scorer is self-contained and runs from an embedded Python script.

**Scope:** prioritize suspected flaky tests for triage.  
Treat scores as heuristics for prioritization, not statistical proof.

---

## Workflow

1. Verify you have run-history data (not just a list of failing tests).
2. Confirm input fields, or normalize from aliases.
3. If required fields are missing, request a corrected file (or create a normalized copy from aliases).
4. Create a temporary directory and run the embedded Python scorer.
5. Run the scorer on your local CSV or JSON file.
6. Read the ranked JSON output and produce a recommendation using:
   - `lower_bound_score`
   - `confidence`
   - `low_data`

The scorer accepts **only local** CSV or JSON files.

If no suitable file is available, ask for a CSV with at least:

- `test_id`
- `result`

It does not fetch CI data.

---

## Input Schema

| field | required | notes |
| --- | --- | --- |
| `test_id` | yes | Unique test identifier. Aliases: `test`, `name`, `testId`, `id`. |
| `result` | yes | Pass/fail outcome. Aliases: `status`, `outcome`. |
| `version` | no | Commit/build/config id; helps separate cross-version changes from instability. |
| `timestamp` | no | ISO-8601 or epoch to order runs within a version. Aliases: `time`, `date`. |

Recognized pass values:  
`pass`, `passed`, `p`, `ok`, `success`, `true`, `1`, `green`

Recognized fail values:  
`fail`, `failed`, `f`, `error`, `failure`, `false`, `0`, `red`

---

## Embedded Python Scorer

```python
#!/usr/bin/env python3
"""Rank flaky tests from CSV or JSON pass/fail run history."""

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from datetime import datetime

PASS_VALUES = {"pass", "passed", "p", "ok", "success", "true", "1", "green"}
FAIL_VALUES = {"fail", "failed", "f", "error", "failure", "false", "0", "red"}


def _normalize_result(raw):
    """Map a raw result cell to True (pass), False (fail), or None."""
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in PASS_VALUES:
        return True
    if value in FAIL_VALUES:
        return False
    return None


def _first_value(row, keys):
    """Return the first aliased value that is present and non-empty.

    Skips missing keys, None, and blank/whitespace strings, but keeps
    other falsy values such as the integer 0 so a test literally named
    "0" (or a "0"/False result cell) is not silently dropped.
    """
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def load_runs(path):
    """Load CSV or JSON rows into normalized test-run dictionaries."""
    if path.lower().endswith(".json"):
        # utf-8-sig transparently strips a leading BOM if present.
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("runs", data.get("results"))
        if not isinstance(data, list):
            raise ValueError(
                "JSON input must be a list of objects or an object with runs/results list"
            )
        rows = data
    else:
        # utf-8-sig strips the BOM that Excel/Sheets prepend to CSV exports,
        # which would otherwise corrupt the first header (e.g. "test_id").
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

    runs = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        test_id = _first_value(row, ("test_id", "test", "name", "testId", "id"))
        result_raw = _first_value(row, ("result", "status", "outcome"))
        if test_id is None or result_raw is None:
            continue

        result = _normalize_result(result_raw)
        if result is None:
            continue

        runs.append(
            {
                "test_id": str(test_id),
                "result": result,
                "version": (
                    str(row["version"])
                    if row.get("version") not in (None, "")
                    else None
                ),
                "timestamp": _first_value(row, ("timestamp", "time", "date")),
                "_order": index,
            }
        )

    if not runs:
        raise ValueError(
            "No usable runs found. Need columns 'test_id' and 'result' with "
            "recognizable pass/fail values."
        )
    return runs


def entropy(results):
    """Return normalized Shannon entropy for pass/fail outcomes."""
    count = len(results)
    if count == 0:
        return 0.0

    p_pass = sum(1 for result in results if result) / count
    p_fail = 1.0 - p_pass
    score = 0.0
    for probability in (p_pass, p_fail):
        if probability > 0:
            score -= probability * math.log(probability, 2)
    return score


def flip_rate(results):
    """Return the fraction of consecutive run pairs that flip pass/fail."""
    count = len(results)
    if count < 2:
        return 0.0
    flips = sum(1 for left, right in zip(results, results[1:]) if left != right)
    return flips / (count - 1)


METRICS = {"entropy": entropy, "flipRate": flip_rate, "fliprate": flip_rate}


def timestamp_key(value):
    """Return a sortable key for numeric, ISO-like, or opaque timestamps."""
    if value in (None, ""):
        return (2, 0, "")

    text = str(value).strip()
    try:
        return (0, float(text), text)
    except ValueError:
        pass

    try:
        return (1, datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp(), text)
    except ValueError:
        return (2, 0, text)


def group_by_test_and_version(runs):
    """Return test_id -> version -> ordered result list."""
    runs_sorted = sorted(
        runs, key=lambda row: (timestamp_key(row["timestamp"]), row["_order"])
    )
    by_test = OrderedDict()
    for run in runs_sorted:
        version = run["version"] if run["version"] is not None else "__all__"
        by_test.setdefault(run["test_id"], OrderedDict()).setdefault(
            version, []
        ).append(run["result"])
    return by_test


def aggregate_unweighted(version_scores):
    """Return a simple average of per-version scores."""
    if not version_scores:
        return 0.0
    return sum(version_scores) / len(version_scores)


def aggregate_weighted(version_scores, lam=0.1):
    """Return an exponentially weighted moving average over versions."""
    if not version_scores:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    count = len(version_scores)
    for index, score in enumerate(version_scores):
        age = (count - 1) - index
        weight = lam * (1 - lam) ** age
        numerator += weight * score
        denominator += weight
    return numerator / denominator if denominator else 0.0


def confidence(total_runs, version_scores):
    """Return confidence in [0, 1] from data volume and score stability."""
    count = max(total_runs, 1)
    data_factor = 1.0 - 1.0 / math.sqrt(count)

    if len(version_scores) >= 2:
        mean = sum(version_scores) / len(version_scores)
        variance = sum((score - mean) ** 2 for score in version_scores) / len(
            version_scores
        )
        stability_factor = max(0.0, 1.0 - math.sqrt(variance))
    else:
        stability_factor = 1.0

    return round(data_factor * stability_factor, 4)


def score_tests(runs, metric="flipRate", model="weighted", lam=0.1, min_reruns=2):
    """Score and rank tests from normalized run dictionaries."""
    metric_fn = METRICS[metric.lower()] if metric.lower() in METRICS else METRICS[metric]
    by_test = group_by_test_and_version(runs)

    results = []
    for test_id, versions in by_test.items():
        per_version = [metric_fn(version_results) for version_results in versions.values()]
        total_runs = sum(len(version_results) for version_results in versions.values())

        if model == "weighted":
            score = aggregate_weighted(per_version, lam=lam)
        else:
            score = aggregate_unweighted(per_version)

        conf = confidence(total_runs, per_version)
        lower_bound = round(max(0.0, score * conf), 4)

        results.append(
            {
                "test_id": test_id,
                "score": round(score, 4),
                "confidence": conf,
                "lower_bound_score": lower_bound,
                "total_runs": total_runs,
                "num_versions": len(per_version),
                "low_data": total_runs < min_reruns,
                "verdict": _verdict(score),
            }
        )

    results.sort(
        key=lambda row: (row["score"], row["confidence"], row["total_runs"]),
        reverse=True,
    )
    for rank, row in enumerate(results, 1):
        row["rank"] = rank
    return results


def _verdict(score):
    """Return a heuristic prioritization band; not a calibrated label."""
    if score <= 0.0:
        return "not_flaky"
    if score < 0.17:
        return "slightly_flaky"
    if score < 0.5:
        return "flaky"
    return "very_flaky"


def write_outputs(results, out_json=None, out_csv=None):
    """Write JSON to stdout or file, and optionally write CSV."""
    payload = json.dumps(results, indent=2)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        print(payload)

    if out_csv:
        fields = [
            "rank",
            "test_id",
            "score",
            "confidence",
            "lower_bound_score",
            "verdict",
            "total_runs",
            "num_versions",
            "low_data",
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow({field: row[field] for field in fields})


def summarize(results):
    """Return a short stderr summary."""
    flaky = [row for row in results if row["score"] > 0]
    very_flaky = [row for row in results if row["verdict"] == "very_flaky"]
    lines = [
        f"Scored {len(results)} tests; {len(flaky)} show some flakiness "
        f"({len(very_flaky)} very flaky).",
    ]

    if flaky:
        lines.append("Top suspects:")
        for row in flaky[:10]:
            flag = "  [LOW DATA]" if row["low_data"] else ""
            lines.append(
                f"  #{row['rank']:<2} {row['test_id']:<40} "
                f"score={row['score']:.3f} conf={row['confidence']:.2f} "
                f"({row['verdict']}){flag}"
            )
    return "\n".join(lines)


def main(argv=None):
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Rank flaky tests from run history.")
    parser.add_argument("input", help="CSV or JSON file of test runs")
    parser.add_argument(
        "--metric",
        default="flipRate",
        choices=["flipRate", "entropy"],
        help="per-version flakiness metric (default: flipRate)",
    )
    parser.add_argument(
        "--model",
        default="weighted",
        choices=["weighted", "unweighted"],
        help="cross-version aggregation (default: weighted)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.1,
        help="EWMA decay for weighted model (default: 0.1; smaller = longer memory)",
    )
    parser.add_argument(
        "--min-reruns",
        type=int,
        default=2,
        help="flag tests with fewer total runs as low-data (default: 2)",
    )
    parser.add_argument("--out", help="write JSON report to this path")
    parser.add_argument("--csv-out", help="also write a CSV report to this path")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the human-readable summary on stderr",
    )
    args = parser.parse_args(argv)

    if not 0 < args.lam <= 1:
        print("error: --lam must be in the range (0, 1]", file=sys.stderr)
        return 1
    if args.min_reruns < 1:
        print("error: --min-reruns must be >= 1", file=sys.stderr)
        return 1

    try:
        runs = load_runs(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results = score_tests(
        runs,
        metric=args.metric,
        model=args.model,
        lam=args.lam,
        min_reruns=args.min_reruns,
    )
    write_outputs(results, out_json=args.out, out_csv=args.csv_out)

    if not args.quiet:
        print("\n" + summarize(results), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Execution

```bash
tmpdir="$(mktemp -d)"
# Copy only the contents of the Python scorer into "$tmpdir/score_flakiness.py".
python3 "$tmpdir/score_flakiness.py" --help

python3 "$tmpdir/score_flakiness.py" runs.csv
python3 "$tmpdir/score_flakiness.py" runs.json \
    --metric entropy --model weighted --lam 0.05 \
    --out report.json --csv-out report.csv
```

Defaults:

- `--metric flipRate`
- `--model weighted`
- `--lam 0.1`
- `--min-reruns 2`

The scorer uses only Python's standard library.

It prints ranked JSON to stdout unless `--out` is provided.  
It prints a short human-readable summary to stderr unless `--quiet` is set.

---

## Output Format

## Summary
Scored `<n>` tests. `<m>` show flakiness. `<k>` need more data.

## Top suspects
| rank | test | score | confidence | lower_bound | verdict | action |
| --- | --- | --- | --- | --- | --- | --- |
| `1` | `example_test` | `0.82` | `0.91` | `0.75` | `very_flaky` | `quarantine` |

## Recommendation
`Fix`, `quarantine`, or `keep watching`.  
Mention low-confidence tests explicitly.

---

## Interpretation

- `score`: observed instability, ranked highest first.
- `confidence`: heuristic support from data volume and score stability.
- `lower_bound_score`: conservative score (score × confidence); prefer this for quarantine/filtering.
- `low_data`: `true` when a test has fewer than `--min-reruns` total runs.
- `verdict`: `not_flaky`, `slightly_flaky`, `flaky`, `very_flaky`.

Report top suspects first and discount low-confidence or low-data rows.
When `low_data: true`, recommend more reruns before quarantine decisions.

If history includes environment/suite/build metadata, compare the ranked output against it manually to separate test-level instability from infrastructure effects.

---

## Notes and Limits

- Scores are meaningful only with representative and sufficient run history.
- `flipRate` needs at least 2 runs per version to be non-zero.
- `entropy` needs at least 2 runs to exceed 0.
- `version`/commit/build metadata helps split cross-version changes from within-version instability.
- This method scores existing history only; it does not run tests, pull CI data, or infer flakiness from one-time failures.
- Never use these scores alone to quarantine when data is sparse or failure patterns can be explained by infrastructure.

This skill is inspired by:

Emily Kowalczyk, Karan Nair, Zebao Gao, Leo Silberstein, Teng Long, Atif Memon.
"Modeling and Ranking Flaky Tests at Apple." ICSE-SEIP 2020. DOI: 10.1145/3377813.3381370
