"""`kva` 命令行:run / score / validate / submit / data / methods。"""
import argparse
import json
import os
import sys

# Override this destination for a separate leaderboard deployment.
SUBMIT_REPO = os.environ.get("KVA_SUBMIT_REPO", "xishi404/KVShareArena")


def _cmd_run(a):
    from .harness import run
    cfg = json.loads(a.method_cfg) if a.method_cfg else None
    run(a.method, a.model, a.track, a.subset, n=a.n, data_dir=a.data_dir, out=a.out, method_cfg=cfg,
        prompt_key=a.prompt_key)


def _cmd_score(a):
    from .data import load_anchors
    from .schema import load
    from .scoring import score_doc
    doc = load(a.result)
    s = doc.get("summary", {})
    track, subset = a.track or s.get("track"), a.subset or s.get("subset")
    if not (track and subset):
        sys.exit("track/subset not in the result file's summary; pass --track/--subset")
    anchors, src = load_anchors(a.model_slug, a.anchors, a.data_dir)
    rep = score_doc(doc, anchors, track, subset, baseline=a.baseline, recompute=not a.trust_self_reported)
    rep["anchors_file"] = src
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=1)


def _cmd_validate(a):
    from .data import load_queryset
    from .schema import load, validate
    doc = load(a.result)
    s = doc.get("summary", {})
    frozen = None
    if s.get("subset") and not a.no_queryset:
        try:
            rows, _ = load_queryset(s["subset"], a.data_dir)
            frozen = [r["_id"] for r in rows]
        except FileNotFoundError as e:
            print(f"[warn] queryset not available for _id check: {e}")
    probs = validate(doc, frozen_ids=frozen, expect_n=None if a.allow_partial else 100)
    if probs:
        print("INVALID:")
        for p in probs:
            print("  -", p)
        sys.exit(1)
    print(f"VALID {a.result} ({s.get('method')} @ {s.get('track')}/{s.get('subset')}, n={s.get('n')})")


def _cmd_submit(a):
    from .data import load_queryset
    from .schema import load, validate
    from .submit import submit
    doc = load(a.result)
    s = doc.get("summary", {})
    frozen = None
    if s.get("subset") and not a.no_queryset:
        try:
            rows, _ = load_queryset(s["subset"], a.data_dir)
            frozen = [r["_id"] for r in rows]
        except FileNotFoundError as e:
            print(f"[warn] queryset not available for _id check: {e}")
    probs = validate(doc, frozen_ids=frozen, expect_n=None if a.allow_partial else 100)
    if probs:
        print("INVALID — 提交被拒,先修下面这些再 submit:")
        for p in probs:
            print("  -", p)
        sys.exit(1)
    print(f"VALID {a.result} ({s.get('method')} @ {s.get('track')}/{s.get('subset')}, n={s.get('n')})")
    submit(a.result, card_path=a.card, repo=a.repo, dry_run=a.dry_run,
           staging_dir=a.staging_dir, branch=a.branch)


def _cmd_data(a):
    from .data import check_manifest, pull, resolve_data_dir
    if a.data_cmd == "pull":
        path, bad = pull(a.repo, a.dest, a.revision)
        print(f"pulled to {path}; manifest: {'no sha256.json' if bad is None else ('OK' if not bad else bad)}")
    elif a.data_cmd == "rebuild-frames":
        from .frames import rebuild
        ids = [x.strip() for x in a.ids.split(",") if x.strip()] if a.ids else None
        rebuild(a.out_dir, ids=ids, split=a.split, snapshot_dir=a.snapshot_dir,
                questions=a.questions, data_dir=a.data_dir, split_file=a.split_file,
                cache_dir=a.cache_dir, verify=not a.no_verify,
                allow_download=not a.no_download)
    else:
        d = resolve_data_dir(a.data_dir)
        bad = check_manifest(d)
        print(f"data_dir={d}; manifest: {'no sha256.json (pre-release local data)' if bad is None else ('OK' if not bad else bad)}")
        if bad:
            sys.exit(1)


def _cmd_methods(a):
    from . import drivers  # noqa: F401
    from .contract import get_method, list_methods
    for m in list_methods():
        c = get_method(m)
        print(f"{m:16s} tracks={sorted(c.track)} class={c.mechanism_class} provenance={c.provenance}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kva", description="KVShareArena harness / scorer")
    sp = ap.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="run a method on a frozen queryset (needs a GPU)")
    r.add_argument("--method", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--track", default="re", choices=["re", "ar"])
    r.add_argument("--subset", required=True)
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--data_dir")
    r.add_argument("--out", required=True)
    r.add_argument("--method_cfg", help="JSON dict passed to the method's setup()")
    r.add_argument("--prompt_key", default="P_c")
    r.set_defaults(fn=_cmd_run)

    s = sp.add_parser("score", help="PGR + paired bootstrap vs free position alignment (CPU)")
    s.add_argument("result")
    s.add_argument("--anchors")
    s.add_argument("--data_dir")
    s.add_argument("--model_slug", default="qwen3-8b")
    s.add_argument("--track")
    s.add_argument("--subset")
    s.add_argument("--baseline", default="naive_pos")
    s.add_argument("--trust_self_reported", action="store_true",
                   help="use stored f1_raw instead of recomputing from pred/golds")
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_score)

    v = sp.add_parser("validate", help="schema / frozen ids / telemetry / method card")
    v.add_argument("result")
    v.add_argument("--data_dir")
    v.add_argument("--allow_partial", action="store_true")
    v.add_argument("--no_queryset", action="store_true")
    v.set_defaults(fn=_cmd_validate)

    sb = sp.add_parser("submit", help="validate, then open a leaderboard pull request (needs gh CLI)")
    sb.add_argument("result")
    sb.add_argument("--card", help="method_card.md; 不给就用结果件里的 method_card 生成一份最小卡片")
    sb.add_argument("--repo", default=SUBMIT_REPO, help="leaderboard repo, e.g. owner/name")
    sb.add_argument("--branch", help="default submit/<method>-<YYYYMMDD>")
    sb.add_argument("--staging_dir", help="default ./.kva_submit/<branch>")
    sb.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                    help="只备文件与 PR 正文并打印手工步骤,不动 git/gh")
    sb.add_argument("--data_dir")
    sb.add_argument("--allow_partial", action="store_true")
    sb.add_argument("--no_queryset", action="store_true")
    sb.set_defaults(fn=_cmd_submit)

    d = sp.add_parser("data", help="pull / check frozen querysets")
    dsp = d.add_subparsers(dest="data_cmd", required=True)
    dp = dsp.add_parser("pull")
    dp.add_argument("--repo", required=True, help="HF dataset repo id")
    dp.add_argument("--dest", default=None)
    dp.add_argument("--revision", default=None)
    dc = dsp.add_parser("check")
    dc.add_argument("--data_dir")
    dr = dsp.add_parser("rebuild-frames",
                        help="rebuild the FRAMES payloads from the official questions and the "
                             "checksum-pinned Wikipedia snapshot, then check every payload "
                             "against the frozen fingerprints (CPU; needs lxml and network)")
    dr.add_argument("--out_dir", required=True)
    dr.add_argument("--split", default="test", choices=["test", "calibration"])
    dr.add_argument("--ids", help="逗号分隔 _id(如 frames_631,frames_490);不给就重建整个 split")
    dr.add_argument("--data_dir", help="装冻结清单的目录(kva data pull 的落点)")
    dr.add_argument("--split_file", help="直接指定 frames_variant/<split>_split.jsonl")
    dr.add_argument("--snapshot_dir", help="快照落点,默认 ~/.cache/kvsharearena/frames_snapshot")
    dr.add_argument("--questions", help="官方 test.tsv 本地路径;不给就从 HF 按 pin 的 revision 取")
    dr.add_argument("--cache_dir", help="物化页面缓存,默认 <out_dir>/page_cache")
    dr.add_argument("--no_verify", action="store_true",
                    help="指纹不符时仍写出 payloads(只用于排查,产出不得用于榜面)")
    dr.add_argument("--no_download", action="store_true",
                    help="不联网:快照与问题集必须已在本地")
    d.set_defaults(fn=_cmd_data)

    m = sp.add_parser("methods", help="list registered methods")
    m.set_defaults(fn=_cmd_methods)

    a = ap.parse_args(argv)
    if a.cmd == "data" and a.data_cmd == "pull" and a.dest is None:
        from .data import DEFAULT_PULL_DIR
        a.dest = DEFAULT_PULL_DIR
    a.fn(a)


if __name__ == "__main__":
    main()
