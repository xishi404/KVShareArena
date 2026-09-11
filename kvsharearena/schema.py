"""kva.result.v1 结果文件 schema 与 `kva validate`。"""
import json

SCHEMA = "kva.result.v1"
SUMMARY_KEYS = ("method", "track", "subset", "model", "n", "backend", "versions", "git_rev")
ITEM_KEYS = ("_id", "pred", "f1_raw", "recomputed_layer_tokens", "dense_payload_layer_tokens",
             "kv_bytes", "ttft_ms")
CARD_KEYS = ("name", "provenance", "backend", "versions", "cost_notes", "training_data")
PROVENANCE = ("official", "port", "reference")


def build_result(summary, results, method_card):
    return {"schema": SCHEMA, "summary": summary, "results": results, "method_card": method_card}


def validate(doc, frozen_ids=None, expect_n=100):
    """返回问题清单(空=通过)。frozen_ids 给出时逐一核对 _id 集合与冻结考卷一致。"""
    problems = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema != {SCHEMA}: {doc.get('schema')!r}")
    s = doc.get("summary") or {}
    for k in SUMMARY_KEYS:
        if k not in s:
            problems.append(f"summary missing {k}")
    rows = doc.get("results")
    if not isinstance(rows, list):
        problems.append("results missing or not a list")
        rows = []
    if s.get("n") != len(rows):
        problems.append(f"summary.n={s.get('n')} != len(results)={len(rows)}")
    if expect_n is not None and len(rows) != expect_n:
        problems.append(f"len(results)={len(rows)} != frozen N={expect_n}")
    ids = [r.get("_id") for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate _id in results")
    if frozen_ids is not None and set(ids) != set(frozen_ids):
        missing = sorted(set(frozen_ids) - set(ids))[:3]
        extra = sorted(set(ids) - set(frozen_ids))[:3]
        problems.append(f"_id set != frozen queryset (missing e.g. {missing}, extra e.g. {extra})")
    for i, r in enumerate(rows):
        for k in ITEM_KEYS:
            if k not in r:
                problems.append(f"results[{i}] missing {k}")
        for k in ("recomputed_layer_tokens", "dense_payload_layer_tokens", "kv_bytes"):
            v = r.get(k)
            if v is not None and (not isinstance(v, (int, float)) or v < 0):
                problems.append(f"results[{i}].{k} not a non-negative number: {v!r}")
        if "golds" not in r:
            problems.append(f"results[{i}] missing golds (scorer recomputes F1 from pred/golds)")
    card = doc.get("method_card") or {}
    for k in CARD_KEYS:
        if k not in card:
            problems.append(f"method_card missing {k}")
    if card.get("provenance") not in PROVENANCE:
        problems.append(f"method_card.provenance must be one of {PROVENANCE}")
    return problems


def load(path):
    with open(path) as f:
        return json.load(f)
