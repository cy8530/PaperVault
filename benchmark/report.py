"""Benchmark report — formatted terminal output + optional JSON export."""
import json


def _bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    filled = int(min(value / max_val, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


# ── Retrieval ─────────────────────────────────────────────

def _print_single_retrieval(s: dict) -> None:
    if not s or s.get("total_questions", 0) == 0:
        return

    print()
    print("─" * 60)
    print("  Retrieval — Single-paper")
    print("─" * 60)
    print(f"  Questions:  {s['total_questions']}")
    print(f"  Found:      {s['found_count']} ({s['found_rate']:.1%})")
    if s.get("mean_rank"):
        print(f"  Mean rank:  {s['mean_rank']:.1f}")
        print(f"  Median rank:{s['median_rank']:.0f}")
    print(f"  MRR:        {s['mrr']:.4f}")
    print()
    print("  Recall@K:")
    for k in (1, 3, 5, 10, 20):
        key = f"Recall@{k}"
        val = s.get(key, 0.0)
        print(f"    @{k:>2}: {val:.3f}  {_bar(val)}")

    # ── Per-difficulty ─────────────────────────────
    by_diff = s.get("by_difficulty", {})
    if by_diff:
        diff_names = {"L1": "Shallow (L1)", "L2": "Medium (L2)", "L3": "Deep (L3)"}
        print()
        print("  By Difficulty:")
        print(f"    {'Level':<16} {'Q':>3}  {'Found':>5}  {'MRR':>6}  {'R@1':>6}  {'R@5':>6}")
        print(f"    {'─' * 50}")
        for label in ("L1", "L2", "L3"):
            info = by_diff.get(label)
            if not info:
                continue
            print(f"    {diff_names[label]:<16} {info['count']:>3}  "
                  f"{info['found']:>5}  {info['MRR']:.4f}  "
                  f"{info['Recall@1']:.4f}  {info['Recall@5']:.4f}")


def _print_cross_retrieval(c: dict) -> None:
    if not c or c.get("total_questions", 0) == 0:
        return

    print()
    print("─" * 60)
    print("  Retrieval — Cross-paper (multi-target)")
    print("─" * 60)
    print(f"  Questions:      {c['total_questions']}")
    print(f"  Targets total:  {c.get('total_target_papers', 0)}")
    print(f"  Found:          {c.get('total_found', 0)} ({c.get('found_rate', 0):.1%})")
    print()
    print("  Multi-target Recall@K (avg fraction of targets in top-K):")
    for k in (1, 3, 5, 10, 20):
        val = c.get(f"Recall@{k}", 0.0)
        print(f"    @{k:>2}: {val:.3f}  {_bar(val)}")


# ── Answer quality ────────────────────────────────────────

_METRIC_LABELS = {
    "faithfulness":      ("Faithfulness", "答案忠实度"),
    "coverage":          ("Coverage",     "信息覆盖率"),
    "context_relevance": ("Context Rel.", "上下文相关性"),
}


def _print_answer_section(results: dict) -> None:
    if not results:
        return

    print()
    print("─" * 60)
    print("  Answer Quality (LLM-as-judge)")
    print("─" * 60)

    # Overall aggregate
    all_avgs = []
    for key, (short, cn) in _METRIC_LABELS.items():
        scores = results.get(key, [])
        valid = [s["score"] for s in scores if isinstance(s, dict) and "score" in s]
        if not valid:
            print(f"  {short:>18} ({cn}): (no data)")
            continue
        avg = sum(valid) / len(valid)
        print(f"  {short:>18} ({cn}): {avg:.3f}  {_bar(avg)}  (n={len(valid)})")
        all_avgs.append(avg)

    if all_avgs:
        composite = sum(all_avgs) / len(all_avgs)
        print(f"  {'─' * 52}")
        print(f"  {'Composite':>18}:  {composite:.3f}  {_bar(composite)}")

    # Per-type + per-difficulty breakdown
    _print_answer_breakdown(results)


def _print_answer_breakdown(results: dict) -> None:
    """Print answer quality broken down by question type and difficulty."""
    # Collect scores by (type, difficulty)
    groups = {}  # key -> list of score dicts

    for metric_key, (short, cn) in _METRIC_LABELS.items():
        scores = results.get(metric_key, [])
        for s in scores:
            if not isinstance(s, dict):
                continue
            qtype = s.get("type", "single")
            diff = s.get("difficulty")
            if qtype == "cross":
                group = "cross"
            elif diff is not None:
                group = f"L{diff}"
            else:
                group = "single"
            if group not in groups:
                groups[group] = {}
            if metric_key not in groups[group]:
                groups[group][metric_key] = []
            groups[group][metric_key].append(s["score"])

    if len(groups) <= 1:
        return

    group_order = ["L1", "L2", "L3", "cross"]
    group_names = {"L1": "Shallow (L1)", "L2": "Medium (L2)",
                   "L3": "Deep (L3)", "cross": "Cross-paper"}
    print()
    print("  By Type / Difficulty:")
    header = f"    {'Group':<18}"
    for key, (short, _) in _METRIC_LABELS.items():
        header += f"  {short:>10}"
    header += "  Composite"
    print(header)
    print(f"    {'─' * 60}")

    for group in group_order:
        if group not in groups:
            continue
        grp = groups[group]
        name = group_names.get(group, group)
        row = f"    {name:<18}"
        comps = []
        for key, (short, _) in _METRIC_LABELS.items():
            vals = grp.get(key, [])
            if vals:
                avg = sum(vals) / len(vals)
                row += f"  {avg:>10.3f}"
                comps.append(avg)
            else:
                row += f"  {'—':>10}"
        if comps:
            row += f"  {sum(comps)/len(comps):.3f}"
        else:
            row += "  —"
        print(row)


# ── Summary ───────────────────────────────────────────────

def print_summary(retrieval: dict, answer: dict) -> None:
    """Print a one-page summary of all benchmark results."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  PaperVault RAG Quality Benchmark".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    s = retrieval.get("single", {})
    c = retrieval.get("cross", {})

    if s.get("total_questions", 0) > 0 or c.get("total_questions", 0) > 0:
        print()
        print("=" * 60)
        print("  Retrieval Quality")
        print("=" * 60)

    _print_single_retrieval(s)
    _print_cross_retrieval(c)

    _print_answer_section(answer)

    if not answer:
        print()
        print("  Note: Run with default 'quality' (no subcommand) to include answer quality.")
    else:
        print()
        print("  Note: Answer quality scores are based on LLM-as-judge")
        print("  (using configured LIGHT_MODEL_ID as the judge).")
    print()


def export_json(retrieval: dict, answer: dict, path: str = "benchmark_report.json") -> None:
    """Export results as JSON."""
    # Strip verbose details for compact export
    def _strip(d):
        if isinstance(d, dict):
            return {k: _strip(v) for k, v in d.items()
                    if k not in ("details", "all_notes", "note_content")}
        return d

    output = {
        "retrieval": {
            "single": _strip(retrieval.get("single", {})),
            "cross": _strip(retrieval.get("cross", {})),
        },
        "answer_quality": _strip(answer),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Report exported to: {path}")
