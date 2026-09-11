"""`kva run`:冻结考卷 × 方法契约 → kva.result.v1。

受体表面(system prompt、LongBench 格式段、chat 模板、贪心解码、max_new_tokens、判分)对所有方法恒定;
方法只在 build_cache/answer 内决定载荷 KV 的出生与修复。build_cache 阶段 harness 不持有 question
(question-blind 由调用序列保证)。
"""
import json
import os
import subprocess
import time

from . import __version__
from .contract import get_method
from .data import exam_scaffold, load_queryset
from .schema import build_result
from .scoring import METRICS


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def _versions():
    out = {"kvsharearena": __version__}
    for mod in ("torch", "transformers", "kvpress"):
        try:
            out[mod] = __import__(mod).__version__
        except AttributeError:                       # 无 __version__ 属性(如 kvpress)→ 退到发行元数据
            try:
                from importlib.metadata import version
                out[mod] = version(mod)
            except Exception:
                pass
        except Exception:
            pass
    return out


def _gemm_libs():
    """进程实际映射的 cuBLAS/cuBLASLt/cuDNN 路径(G-42:cublasLt 版本不同会造成 ~7% 样本级分叉)。"""
    try:
        with open("/proc/self/maps") as f:
            return sorted({l.split()[-1] for l in f if "cublas" in l or "cudnn" in l})
    except Exception:
        return None


def default_method_card(M, backend):
    return {"name": M.name, "provenance": getattr(M, "provenance", "port"),
            "mechanism_class": getattr(M, "mechanism_class", 0),
            "backend": backend, "versions": _versions(),
            "training_data": getattr(M, "training_data", None),
            "cost_notes": getattr(M, "cost_notes", "recomputed/dense payload layer-tokens counted on tensors; "
                                                    "kv_bytes = held payload KV tensor bytes; ttft_ms host-side to first token")}


def _log_flush(msg):
    print(msg, flush=True)                       # sbatch 下 stdout 是块缓冲,不 flush 就看不到逐题进度


def run(method_name, model_id, track, subset, n=100, data_dir=None, out=None, method_cfg=None,
        prompt_key="P_c", log=_log_flush):
    if track != "re":
        raise NotImplementedError("v0.1 `kva run` covers track 're' (Retrieved Evidence) only; "
                                  "track 'ar' rows are scored from released per-sample outputs via `kva score`")
    from . import drivers  # noqa: F401  注册内置 driver
    data, src = load_queryset(subset, data_dir, n=n)
    exam = exam_scaffold(subset, track, prompt_key)
    cfg = {"exam": exam, **(method_cfg or {})}
    M = get_method(method_name)()
    if track not in set(M.track):
        raise ValueError(f"method {method_name} declares tracks {sorted(M.track)}, not {track!r}")
    M.setup(model_id, cfg)
    backend = getattr(M, "backend", "hf")
    log(f"method={method_name} model={model_id} track={track} subset={subset} n={len(data)} "
        f"max_new={exam['max_new_tokens']} queryset={src['file']} sha256={src['sha256'][:12]}")
    versions, git_rev = _versions(), _git_rev()   # 循环前取:结尾任何一步卡死都不该吞掉整轮生成
    results = []
    t_all = time.time()
    partial = f"{out}.partial" if out else None
    for i, d in enumerate(data):
        q, golds, chunks = d["input"], d["answers"], d["chunks"]
        caches = [M.build_cache(ch) for ch in chunks]
        res = M.answer(q, caches, cfg)
        pred = res.text
        rec = {"_id": d.get("_id"), "golds": golds, "n_chunks": len(chunks), "pred": pred,
               **{m: fn(pred, golds) for m, fn in METRICS.items()},
               "think_leak": int("<think>" in pred),
               "recomputed_layer_tokens": int(res.recomputed_layer_tokens),
               "dense_payload_layer_tokens": int(res.dense_payload_layer_tokens),
               "kv_bytes": int(res.kv_bytes), "ttft_ms": res.ttft_ms, "aux": res.aux}
        results.append(rec)
        log(f"[{i+1}/{len(data)}] f1={rec['f1_raw']:.2f} ttft={res.ttft_ms if res.ttft_ms is None else round(res.ttft_ms)}ms")
        if partial and (i + 1) % 10 == 0:
            os.makedirs(os.path.dirname(os.path.abspath(partial)), exist_ok=True)
            with open(partial, "w") as f:
                json.dump({"schema": "kva.result.v1.partial", "n_done": i + 1, "results": results}, f,
                          ensure_ascii=False)
    M.teardown()
    n_ = len(results)
    summary = {"method": method_name, "track": track, "subset": subset, "model": model_id, "n": n_,
               "backend": backend, "versions": versions, "git_rev": git_rev, "gemm_libs": _gemm_libs(),
               "prompt_key": prompt_key, "max_new_tokens": exam["max_new_tokens"],
               "queryset": {"file": os.path.basename(src["file"]), "sha256": src["sha256"],
                            "manifest_checked": src["manifest_checked"]},
               "sec_total": round(time.time() - t_all, 1),
               **{m: sum(r[m] for r in results) / n_ for m in METRICS},
               "think_leak_rate": sum(r["think_leak"] for r in results) / n_,
               "recompute_fraction": (sum(r["recomputed_layer_tokens"] for r in results) /
                                      max(1, sum(r["dense_payload_layer_tokens"] for r in results)))}
    doc = build_result(summary, results, default_method_card(M, backend))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        log(f"wrote {out}")
    return doc
