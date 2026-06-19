#!/usr/bin/env python3
"""PaperVault RAG quality benchmark.

Usage:
    python benchmark.py                    # quality (all metrics, 5 random papers)
    python benchmark.py quality            # all quality metrics
    python benchmark.py quality retrieval  # retrieval only (zero LLM cost)
    python benchmark.py quality answer     # answer quality only
    python benchmark.py gen                # generate test data only
    python benchmark.py quality --papers 10 --json
    python benchmark.py quality --detail 2  # specific RAG detail level
    python benchmark.py quality --no-cache # force re-generation

First run generates test questions from sampled papers and caches them
as ``benchmark/data/cache/*.json``.  Subsequent runs use the cache.
"""
import argparse
import random
import sys
from pathlib import Path

# Add project root to path for direct execution
_PROJ_ROOT = Path(__file__).parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from benchmark.data.generate import build_test_data
from benchmark.metrics.retrieval import evaluate_retrieval
from benchmark.metrics.judge import (
    evaluate_faithfulness, evaluate_coverage, evaluate_context_relevance,
    get_rag_context,
)
from benchmark.report import print_summary, export_json


def run_quality(args):
    """Run quality benchmark."""
    # Normalize metric from positional or --metrics flag
    args.metrics = getattr(args, "metric2", None) or args.metric

    print(f"PaperVault RAG Quality Benchmark")
    print(f"  Sampling {args.papers} papers, {args.questions_per_paper} Q/paper")
    print(f"  Cross-paper questions: {args.cross_paper}")
    print(f"  Seed: {args.seed}")
    random.seed(args.seed)

    # ── Build test data ─────────────────────────────────
    print(f"\n  [1/4] Generating test data...")
    test_data = build_test_data(
        n_papers=args.papers,
        questions_per_paper=args.questions_per_paper,
        cross_paper_questions=args.cross_paper,
        use_cache=not args.no_cache,
        use_cache_ref=not getattr(args, "no_cache_ref", False),
    )
    single_count = len(test_data["single"])
    cross_count = len(test_data["cross"])
    n_papers = len(test_data["papers"])
    print(f"  Single-paper: {single_count} Q ({n_papers} papers)")
    if test_data["single"]:
        by_d = {1: 0, 2: 0, 3: 0}
        for item in test_data["single"]:
            by_d[item["difficulty"]] = by_d.get(item["difficulty"], 0) + 1
        print(f"    L1 (shallow): {by_d[1]}, L2 (medium): {by_d[2]}, L3 (deep): {by_d[3]}")
    print(f"  Cross-paper:  {cross_count} Q")

    retrieval_results = {}
    answer_results = {}

    if args.metrics in ("all", "retrieval"):
        print(f"\n  [2/4] Evaluating retrieval quality...")
        retrieval_results = evaluate_retrieval(test_data, progress=print_status)

        s = retrieval_results.get("single", {})
        print(f"  Single-paper: MRR={s.get('mrr', 0):.4f}  "
              f"Recall@5={s.get('Recall@5', 0):.3f}  "
              f"Found={s.get('found_count', 0)}/{s.get('total_questions', 0)}")
        # Per-difficulty
        for label in ("L1", "L2", "L3"):
            info = s.get("by_difficulty", {}).get(label)
            if info:
                print(f"    {label}: MRR={info['MRR']:.4f}  "
                      f"Recall@5={info['Recall@5']:.3f}  ({info['count']}Q)")

        c = retrieval_results.get("cross", {})
        if c.get("total_questions", 0) > 0:
            print(f"  Cross-paper:  Recall@5={c.get('Recall@5', 0):.3f}  "
                  f"Found={c.get('total_found', 0)}/{c.get('total_target_papers', 0)} targets")

    if args.metrics in ("all", "answer"):
        # Sample: single questions (stratified by difficulty) + all cross-paper Qs
        single_by_diff = {1: [], 2: [], 3: []}
        for item in test_data["single"]:
            single_by_diff[item["difficulty"]].append(item)

        answer_sample = []
        for diff in (1, 2, 3):
            sample_n = min(len(single_by_diff[diff]), 2)  # up to 2 per difficulty
            answer_sample.extend(single_by_diff[diff][:sample_n])

        # Add cross-paper questions
        answer_sample.extend(test_data["cross"])

        faithfulness_scores = []
        coverage_scores = []
        relevance_scores = []

        print(f"\n  [3/4] Evaluating answer quality ({len(answer_sample)} questions)...")
        for i, item in enumerate(answer_sample):
            q = item["question"]
            ref = item.get("reference_answer", "")
            qtype = item.get("type", "single")
            diff_label = f" L{item['difficulty']}" if "difficulty" in item else ""
            tag = f"[{qtype}{diff_label}]"

            if not ref:
                continue

            print_status(f"Q [{i+1}/{len(answer_sample)}] {tag}: {q[:50]}...")

            try:
                answer, note_contexts, chunk_contexts = get_rag_context(
                    q, n_papers=args.papers, detail=args.detail)
                all_contexts = [c for c in note_contexts + chunk_contexts if c.strip()]

                if not answer:
                    continue

                # Judge: faithfulness
                f = evaluate_faithfulness(answer, all_contexts) if all_contexts else {"score": 0}
                faithfulness_scores.append({
                    "question": q, "type": qtype, "difficulty": item.get("difficulty"),
                    "paper_id": item.get("paper_id", ""), **f})

                # Judge: coverage vs reference
                c = evaluate_coverage(answer, ref)
                coverage_scores.append({
                    "question": q, "type": qtype, "difficulty": item.get("difficulty"),
                    "paper_id": item.get("paper_id", ""), **c})

                # Judge: context relevance
                r = evaluate_context_relevance(q, all_contexts) if all_contexts else {"score": 0}
                relevance_scores.append({
                    "question": q, "type": qtype, "difficulty": item.get("difficulty"),
                    "paper_id": item.get("paper_id", ""), **r})

            except Exception as e:
                print(f"    [SKIP] Error: {e}")
                continue

        answer_results = {
            "faithfulness": faithfulness_scores,
            "coverage": coverage_scores,
            "context_relevance": relevance_scores,
        }

    # ── Report ──────────────────────────────────────────
    print(f"\n  [4/4] Report")
    print_summary(retrieval_results, answer_results)

    if args.json:
        export_json(retrieval_results, answer_results)

    return retrieval_results, answer_results


def run_gen(args):
    """Generate test data only (no evaluation)."""
    test_data = build_test_data(
        n_papers=args.papers,
        questions_per_paper=args.questions_per_paper,
        cross_paper_questions=args.cross_paper,
        use_cache=not args.no_cache,
        use_cache_ref=not getattr(args, "no_cache_ref", False),
    )
    print(f"Single-paper questions: {len(test_data['single'])}")
    for item in test_data["single"]:
        print(f"  [L{item['difficulty']}] [{item['paper_id'][:40]}] {item['question']}")
    print(f"Cross-paper questions: {len(test_data['cross'])}")
    for item in test_data["cross"]:
        print(f"  [cross] {item['question']}")


def print_status(msg):
    print(f"    {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="PaperVault RAG quality benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                        # all metrics, 5 papers
  python benchmark.py quality retrieval      # retrieval only (zero cost)
  python benchmark.py quality answer         # answer quality only
  python benchmark.py gen                    # generate test data only
  python benchmark.py quality --papers 10 --detail 2 --json
  python benchmark.py quality --cross-paper 5 --no-cache
""",
    )

    sub = parser.add_subparsers(dest="command")

    # quality (default)
    q = sub.add_parser("quality", help="Run quality evaluation (default)")
    q.add_argument("metric", nargs="?", choices=["all", "retrieval", "answer"],
                   default="all",
                   help="Which metrics to evaluate (default: all)")
    q.add_argument("--metrics", dest="metric2", choices=["all", "retrieval", "answer"],
                   help="Same as positional")
    q.add_argument("--papers", type=int, default=5,
                   help="Number of papers to sample (default: 5)")
    q.add_argument("--questions-per-paper", type=int, default=5,
                   help="Single-paper questions per paper (default: 5)")
    q.add_argument("--cross-paper", type=int, default=5,
                   help="Cross-paper survey questions (default: 5, set 0 to disable)")
    q.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    q.add_argument("--no-cache", action="store_true",
                   help="Force re-generation of ALL test data (questions + reference answers)")
    q.add_argument("--no-cache-ref", action="store_true",
                   help="Force re-generation of reference answers only (keep cached questions)")
    q.add_argument("--json", action="store_true",
                   help="Export results as benchmark_report.json")
    q.add_argument("--detail", choices=["auto", "1", "2", "3", "all"],
                   default="auto",
                   help="RAG detail level for answer evaluation (default: auto)")
    q.set_defaults(func=run_quality)

    # gen
    g = sub.add_parser("gen", help="Generate test data only")
    g.add_argument("--papers", type=int, default=5)
    g.add_argument("--questions-per-paper", type=int, default=5)
    g.add_argument("--cross-paper", type=int, default=5)
    g.add_argument("--no-cache", action="store_true",
                   help="Force re-generation of ALL test data")
    g.add_argument("--no-cache-ref", action="store_true",
                   help="Force re-generation of reference answers only")
    g.set_defaults(func=run_gen)

    args = parser.parse_args()

    if args.command is None:
        args = parser.parse_args(["quality"])

    args.func(args)


if __name__ == "__main__":
    main()
