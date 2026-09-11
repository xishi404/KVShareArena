"""冻结考卷、锚点与 prompt 骨架的读取;`kva data pull/check`。

数据目录解析顺序:显式 --data_dir > 环境变量 KVA_DATA_DIR > ~/.cache/kvsharearena/data(pull 的落点)
。锚点解析顺序:显式 --anchors > <data_dir>/anchors/<file>
> 包内 assets/anchors/<file>(随包分发的副本,与 platform/anchors/ 同源同 SHA)。
"""
import hashlib
import json
import os
from importlib import resources

DEFAULT_PULL_DIR = os.path.expanduser("~/.cache/kvsharearena/data")
SUBSETS_RE = ("qasper", "multifieldqa_en", "hotpotqa")
FROZEN_N = 100


def resolve_data_dir(explicit=None):
    for cand in (explicit, os.environ.get("KVA_DATA_DIR"), DEFAULT_PULL_DIR):
        if cand and os.path.isdir(cand):
            return cand
    raise FileNotFoundError("no queryset directory found; run `kva data pull --repo <hf dataset id>` "
                            "or set KVA_DATA_DIR / --data_dir")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def check_manifest(data_dir):
    """<data_dir>/sha256.json = {relpath: sha256}(HF 数据页随考卷发布)。缺该文件时返回 None(预发布本地数据)。"""
    mf = os.path.join(data_dir, "sha256.json")
    if not os.path.exists(mf):
        return None
    with open(mf, encoding="utf-8") as handle:
        expect = json.load(handle)
    bad = {}
    for rel, sha in expect.items():
        p = os.path.join(data_dir, rel)
        got = sha256_file(p) if os.path.exists(p) else None
        if got != sha:
            bad[rel] = {"expected": sha, "got": got}
    return bad


def load_queryset(subset, data_dir=None, n=FROZEN_N, strict_manifest=True):
    d = resolve_data_dir(data_dir)
    bad = check_manifest(d)
    if bad:
        raise RuntimeError(f"queryset manifest mismatch in {d}: {bad}")
    path = os.path.join(d, f"{subset}_n{FROZEN_N}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return rows[:n], {"data_dir": d, "file": path, "sha256": sha256_file(path),
                      "manifest_checked": bad is not None}


def _asset(name):
    return resources.files("kvsharearena").joinpath("assets", name)


def load_prompts():
    return json.loads(_asset("prompts.json").read_text(encoding="utf-8"))


def load_longbench_config():
    return json.loads(_asset("longbench_config.json").read_text(encoding="utf-8"))


def exam_scaffold(subset, track="re", prompt_key="P_c"):
    """冻结考卷的 prompt 骨架:system prompt(消费侧角色)+ LongBench 官方格式段三切 + 解码上限。"""
    P = load_prompts()
    cfg = load_longbench_config()
    fmt = cfg["dataset2prompt"][subset]
    pre_ctx, rest = fmt.split("{context}", 1)
    mid_q, tail = rest.split("{input}", 1)
    return {"track": track, "subset": subset, "prompt_key": prompt_key,
            "system_prompt": P[prompt_key], "pre_ctx": pre_ctx, "mid_q": mid_q, "tail": tail,
            "max_new_tokens": int(cfg["dataset2maxlen"][subset])}


def load_anchors(model_slug="qwen3-8b", explicit=None, data_dir=None):
    fname = f"anchors_{model_slug}.json"
    cands = [explicit]
    for dd in (data_dir, os.environ.get("KVA_DATA_DIR"), DEFAULT_PULL_DIR):
        if dd:
            cands.append(os.path.join(dd, "anchors", fname))
    for c in cands:
        if c and os.path.exists(c):
            return json.load(open(c)), c
    a = _asset(os.path.join("anchors", fname))
    if a.is_file():
        return json.loads(a.read_text(encoding="utf-8")), str(a)
    raise FileNotFoundError(f"anchors file {fname} not found (looked at {[c for c in cands if c]} and package assets)")


def pull(repo_id, dest=DEFAULT_PULL_DIR, revision=None):
    from huggingface_hub import snapshot_download
    os.makedirs(dest, exist_ok=True)
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest, revision=revision)
    bad = check_manifest(path)
    return path, bad
