"""Retrieval quality metrics — Recall@K, MRR (local computation, zero LLM cost).

Supports:
  - Single-paper questions: one target paper_id, rank-based recall
  - Cross-paper questions: multiple relevant papers, intersection-based recall
  - Per-difficulty breakdown for single-paper questions
"""
from paper_vault.indexer.store import search_notes
from paper_vault.indexer.embedder import embed_texts


def evaluate_retrieval(test_data: dict, progress=None) -> dict:
    """Evaluate retrieval for single-paper and cross-paper questions.

    Args:
        test_data: dict from build_test_data() with "single" and "cross" keys.

    Returns:
        dict with overall + per-difficulty + per-type metrics.
    """
    single_items = test_data.get("single", [])
    cross_items = test_data.get("cross", [])

    # ── Single-paper evaluation ─────────────────────────
    single_details = []
    single_ranks = []
    by_difficulty = {1: {"ranks": [], "total": 0},
                     2: {"ranks": [], "total": 0},
                     3: {"ranks": [], "total": 0}}

    for i, item in enumerate(single_items):
        if progress:
            progress(f"Retrieval [{i+1}/{len(single_items)}] "
                     f"(L{item['difficulty']}): {item['question'][:50]}...")

        q_vec = embed_texts([item["question"]], is_query=True)[0]
        results = search_notes(q_vec, top_k=20)

        rank = None
        for j, r in enumerate(results):
            if r["paper_id"] == item["paper_id"]:
                rank = j + 1
                break

        detail = {
            "paper_id": item["paper_id"],
            "question": item["question"],
            "difficulty": item["difficulty"],
            "rank": rank,
            "found": rank is not None,
        }
        single_details.append(detail)

        diff = item["difficulty"]
        by_difficulty[diff]["total"] += 1
        if rank is not None:
            single_ranks.append(rank)
            by_difficulty[diff]["ranks"].append(rank)

    # ── Cross-paper evaluation (multi-target) ───────────
    cross_details = []
    cross_recall_at_k = {}

    for i, item in enumerate(cross_items):
        if progress:
            progress(f"Cross-retrieval [{i+1}/{len(cross_items)}]: "
                     f"{item['question'][:50]}...")

        q_vec = embed_texts([item["question"]], is_query=True)[0]
        results = search_notes(q_vec, top_k=20)

        target_ids = set(item.get("relevant_paper_ids", []))
        found_ids = set()
        ranks = {}
        for j, r in enumerate(results):
            pid = r["paper_id"]
            if pid in target_ids:
                found_ids.add(pid)
                if pid not in ranks:
                    ranks[pid] = j + 1

        detail = {
            "question": item["question"],
            "target_count": len(target_ids),
            "found_ids": list(found_ids),
            "missed_ids": list(target_ids - found_ids),
            "ranks": ranks,
        }
        cross_details.append(detail)
        # Store for aggregate computation
        if "cross_ranks" not in cross_recall_at_k:
            cross_recall_at_k["cross_ranks"] = []
        cross_recall_at_k["cross_ranks"].append(detail)

    # ── Aggregate metrics ───────────────────────────────
    return {
        "single": _compute_single_metrics(single_items, single_ranks, single_details,
                                           by_difficulty),
        "cross": _compute_cross_metrics(cross_items, cross_details),
    }


def _compute_single_metrics(items, ranks, details, by_diff):
    total = len(items)
    found = len(ranks)

    recall = {}
    for k in (1, 3, 5, 10, 20):
        recall[f"Recall@{k}"] = (sum(1 for r in ranks if r <= k) / total) if total > 0 else 0.0

    mrr = (sum(1.0 / r for r in ranks) / total) if total > 0 else 0.0

    result = {
        "total_questions": total,
        "found_count": found,
        "found_rate": found / total if total > 0 else 0.0,
        "mean_rank": sum(ranks) / len(ranks) if ranks else None,
        "median_rank": _median(ranks) if ranks else None,
        "mrr": round(mrr, 4),
        **{k: round(v, 4) for k, v in recall.items()},
        "details": details,
    }

    # Per-difficulty breakdown
    diff_summary = {}
    for diff in (1, 2, 3):
        d = by_diff[diff]
        t = d["total"]
        r = d["ranks"]
        if t == 0:
            continue
        dr = {}
        for k in (1, 3, 5, 10, 20):
            dr[f"Recall@{k}"] = (sum(1 for rank in r if rank <= k) / t)
        dr["MRR"] = (sum(1.0 / rank for rank in r) / t) if r else 0.0
        dr["count"] = t
        dr["found"] = len(r)
        diff_summary[f"L{diff}"] = {k: round(v, 4) for k, v in dr.items()}

    result["by_difficulty"] = diff_summary
    return result


def _compute_cross_metrics(items, details):
    if not items or not details:
        return {"total_questions": 0, "details": []}

    # Aggregate: average fraction of targets found at each K
    ks = (1, 3, 5, 10, 20)
    recall = {f"Recall@{k}": [] for k in ks}
    total_targets = 0
    total_found = 0

    for d in details:
        tc = d["target_count"]
        total_targets += tc
        for k in ks:
            found_in_k = sum(1 for rank in d["ranks"].values() if rank <= k)
            recall[f"Recall@{k}"].append(found_in_k / tc if tc > 0 else 0.0)
        total_found += len(d["found_ids"])

    result = {
        "total_questions": len(items),
        "total_target_papers": total_targets,
        "total_found": total_found,
        "found_rate": total_found / total_targets if total_targets > 0 else 0.0,
        "details": details,
    }
    for k in ks:
        vals = recall[f"Recall@{k}"]
        result[f"Recall@{k}"] = (sum(vals) / len(vals)) if vals else 0.0

    return result


def _median(values: list[int]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0
