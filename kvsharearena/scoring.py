"""判分器(纯 CPU):LongBench 官方词面 F1 + PGR + 对免费位置修复的配对 bootstrap。

- 词面指标函数逐字复用 pilot/pilot_naive_reuse.py(源自 LongBench 官方 qa_f1_score 实现)。
- paired() 逐字复用 pilot/k121_np_allrows.py(K107 协议:seed 20260730,B=10000,
  boots[249]/boots[9749] 为 95% CI,闭区间含 0 判 n.s.,精确双侧符号检验)。
- PGR 定义同 EXP_SETTING §8 V34:PGR=(method−floor)/(oracle−floor);拼接列 floor/oracle=K52 锚点两分支,
  笔记列 floor=K52 hotpotqa floor、oracle=笔记列 t2t;oracle−floor≤1e-3 时 PGR 无定义。
"""
import random
import re
import string
from collections import Counter
from math import comb

SEED, B = 20260730, 10000


# ---------- 词面指标(LongBench 官方实现) ----------
def normalize_answer(s):
    def rm_art(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def rm_punc(t):
        return "".join(c for c in t if c not in set(string.punctuation))

    return " ".join(rm_art(rm_punc(s.lower())).split())


def _f1(pred, gt):
    p, g = normalize_answer(pred).split(), normalize_answer(gt).split()
    ns = sum((Counter(p) & Counter(g)).values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def qa_f1(pred, golds):
    return max((_f1(pred, g) for g in golds), default=0.0)


def first_line(s):
    t = s.strip().split("\n")[0].strip()
    return re.sub(r"^(the\s+)?(answer|final answer)\s*(is|:)\s*", "", t, flags=re.I).strip()


def qa_f1_first(pred, golds):
    return qa_f1(first_line(pred), golds)


def contains(pred, golds):
    np_ = normalize_answer(pred)
    return 1.0 if any(normalize_answer(g) and normalize_answer(g) in np_ for g in golds) else 0.0


METRICS = {"f1_raw": qa_f1, "f1_first": qa_f1_first, "contains": contains}


# ---------- PGR ----------
def pgr(method_mean, floor_mean, oracle_mean):
    return None if oracle_mean - floor_mean <= 1e-3 else round((method_mean - floor_mean) / (oracle_mean - floor_mean), 4)


def mean_of(fm):
    vs = [v for v in fm.values() if v is not None]
    if not vs:
        raise ValueError("empty score set")
    return sum(vs) / len(vs)


# ---------- 配对 bootstrap(K107 协议) ----------
def paired(avm, bvm, label):
    ids_common = sorted(set(avm) & set(bvm))
    ids_common_scorable = [i for i in ids_common if avm[i] is not None and bvm[i] is not None]
    unmatched = (len(avm) - len(ids_common)) + (len(bvm) - len(ids_common))
    n_none = len(ids_common) - len(ids_common_scorable)
    diff = [avm[i] - bvm[i] for i in ids_common_scorable]
    n = len(diff)
    result = {"label": label, "n_a": len(avm), "n_b": len(bvm),
              "n_common_ids": len(ids_common), "n_unmatched_ids": unmatched,
              "n_common_but_unscorable": n_none, "n_paired_scored": n}
    if n == 0:
        result.update({"mean_diff": None, "ci95": None, "ci_contains_zero": None,
                       "sign_test_p": None, "pos/neg/tie": None,
                       "note": "n_paired_scored=0,统计量置空"})
        return result
    mean = sum(diff) / n
    rng = random.Random(SEED)
    boots = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    lo, hi = boots[249], boots[9749]
    pos = sum(1 for x in diff if x > 1e-12)
    neg = sum(1 for x in diff if x < -1e-12)
    m = pos + neg
    sp = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(min(pos, neg) + 1)) / 2 ** m)
    result.update({"mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                   "ci_contains_zero": lo <= 0 <= hi, "sign_test_p": round(sp, 4),
                   "pos/neg/tie": f"{pos}/{neg}/{n - pos - neg}"})
    if unmatched or n_none:
        result["note"] = (f"{unmatched} 个 _id 未能在两侧同时出现(如实报,未丢弃到统计外);"
                          f"另有 {n_none} 个 _id 双侧都在但至少一侧 f1_raw 缺失(同样排除出 diff)")
    return result


# ---------- 整份结果文件判分 ----------
def item_scores(doc):
    """{_id: f1_raw}。既吃 kva.result.v1,也吃历史 {summary, results} 存盘件。"""
    rows = doc["results"]
    out = {}
    for r in rows:
        if r["_id"] in out:
            raise ValueError(f"duplicate _id {r['_id']}")
        out[r["_id"]] = r.get("f1_raw")
    return out


def rescore_items(doc):
    """从 pred/golds 重新算词面分(判分器不信任自报 f1)。返回 {_id: f1_raw};缺 pred/golds 的条目报错。"""
    out = {}
    for r in doc["results"]:
        out[r["_id"]] = qa_f1(r["pred"], r["golds"])
    return out


def score_doc(doc, anchors, track, subset, baseline="naive_pos", recompute=True):
    """anchors[track][subset] = {"floor": {id:f1}, "oracle": {id:f1}, "naive_pos": {id:f1}, ...}。"""
    anc = anchors["tracks"][track][subset]
    items = rescore_items(doc) if recompute else item_scores(doc)
    stored = item_scores(doc)
    drift = sum(1 for i in items if stored.get(i) is not None and abs(stored[i] - items[i]) > 1e-9)
    floor_m, oracle_m = mean_of(anc["floor"]), mean_of(anc["oracle"])
    m = mean_of(items)
    out = {"track": track, "subset": subset, "n": len(items),
           "f1_mean": round(m, 4), "floor_f1": round(floor_m, 4), "oracle_f1": round(oracle_m, 4),
           "pgr": pgr(m, floor_m, oracle_m),
           "self_reported_f1_drift_items": drift,
           "anchor_sources": anc.get("sources", {})}
    if baseline in anc:
        out[f"vs_{baseline}"] = paired(items, anc[baseline], f"{doc.get('summary', {}).get('method', '?')} vs {baseline} @{track}/{subset}")
        out[f"{baseline}_f1"] = round(mean_of(anc[baseline]), 4)
        out[f"{baseline}_pgr"] = pgr(mean_of(anc[baseline]), floor_m, oracle_m)
    return out
