"""FRAMES 载荷重建器:从官方问题集 + checksum 钉死的 Wikipedia 快照复原每条样本的正文,
并逐条与冻结的载荷指纹核对。指纹对不上就拒绝交付载荷。

为什么要有这个模块:HF 数据页的 `frames_variant` 子板只发「样本 id + 载荷指纹」,不发
Wikipedia 正文(许可 CC BY-SA + 体积 173MB)。拿到包的人用

    kva data rebuild-frames --split test --out_dir <dir>

在本地把正文重建出来,重建结果逐条与我们冻结的指纹比对,一致才写出 payloads。

两个上游来源,都按 checksum 钉死:
  1. 问题/答案/wiki 链接:官方 `google/frames-benchmark` 的 `test.tsv`,revision 钉死,
     文件 sha256 钉死。
  2. 页面正文:MLCommons 托管的 MLPerf E2E-RAG FRAMES artifact(`docs.tar.gz` +
     `url_mapping.json`,md5 与 sha256 双钉)。这是 **2026 抓取的 HTML 快照**,不是 FRAMES
     作者当年的 2024 快照——我们全程把它标注为 dataset variant。

正文的 HTML→text 物化配方与指纹算法都是**逐字搬运**自当年的两个脚本,不是重写:
  * 物化 / URL 归一 / 问题归一:`pilot/k95_frames_mlperf_audit.py`
    (vendored at sha256 417f53f1521b8fed699393e13a2a42a38a7e204fa746721b769a2dbf08ae2fd9)
  * 指纹(source/bundle/question)与载荷 schema:`pilot/k96_build_queryset.py`
    (vendored at sha256 925f6e12d0adb88b51d192fd2d2fe1b8c003fb23a73750c8580f9adaba83011c)
搬运而非 import,是因为 `pilot/` 不随包分发;`vendor_provenance()` 会在 pilot 脚本还在盘上时
比对它们现在的 sha256,漂移了就在报告里记一条 warning。

需要 `lxml`(`pip install kvsharearena[frames]`)。纯 CPU,不需要 GPU / tokenizer。
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import tarfile
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- pins

# 官方 FRAMES release(问题、答案、wiki 链接)
QUESTIONS_REPO = "google/frames-benchmark"
QUESTIONS_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
QUESTIONS_FILENAME = "test.tsv"
QUESTIONS_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"
QUESTIONS_HTTP = (
    f"https://huggingface.co/datasets/{QUESTIONS_REPO}/resolve/"
    f"{QUESTIONS_REVISION}/{QUESTIONS_FILENAME}"
)

# MLPerf E2E-RAG FRAMES artifact(页面 HTML 快照),MLCommons R2 托管
SNAPSHOT_BASE_URL = "https://inference.mlcommons-storage.org/frames-benchmark-dataset"
SNAPSHOT_MD5_MANIFEST = (
    "https://inference.mlcommons-storage.org/metadata/frames-benchmark-dataset.md5"
)
ARCHIVE_NAME = "docs.tar.gz"
ARCHIVE_MD5 = "994551fd1a8455f22fa3b5f68f0d8d40"
ARCHIVE_SHA256 = "f5e6d7c14f93cdd0af49f9f72e2311419af7c9b0634755fb2310058a84e36e80"
MAPPING_NAME = "url_mapping.json"
MAPPING_MD5 = "abe078da480c5f3014f7f375dc9d396c"
MAPPING_SHA256 = "7992a39c3b7a647d9e4e64896149e724b878f13e134d158d6b2b02f3d4d94c54"
ATTRIBUTION_NAMES = ("NOTICE", "LICENSE")
USER_AGENT = "kvsharearena/0.1 (+https://huggingface.co/datasets)"

SNAPSHOT_VARIANT = "mlperf-e2e-rag-frames-html-2026"
CACHE_SCHEMA = "k95.frames.materialized-page.v1"

DEFAULT_SNAPSHOT_DIR = os.path.expanduser("~/.cache/kvsharearena/frames_snapshot")

VENDORED_FROM = {
    "pilot/k95_frames_mlperf_audit.py":
        "417f53f1521b8fed699393e13a2a42a38a7e204fa746721b769a2dbf08ae2fd9",
    "pilot/k96_build_queryset.py":
        "925f6e12d0adb88b51d192fd2d2fe1b8c003fb23a73750c8580f9adaba83011c",
}

REFERENCE_SECTION_NAMES = {
    "references",
    "external links",
    "further reading",
    "bibliography",
    "sources",
    "citations",
    "notes",
}


# ------------------------------------------------- hashing / normalization
# 以下到 materialize_html 为止,逐字搬自 pilot/k95_frames_mlperf_audit.py。

def hash_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_question(value: str) -> str:
    return normalize_text(value).lower()


def canonicalize_url(value: str) -> tuple[str, list[str]]:
    """Normalize documented URL variants without resolving missing pages live."""
    original = normalize_text(value).strip("'\"")
    value = original
    changes = []
    cleaned = re.sub(
        r"\s*\(NOT REQUIRED, BUT HELPFUL\)\s*$", "", value, flags=re.IGNORECASE
    )
    if cleaned != value:
        value = cleaned
        changes.append("removed_optional_annotation")
    if value.endswith(","):
        value = value[:-1]
        changes.append("removed_terminal_comma")
    if value.startswith("en.wikipedia.org/"):
        value = "https://" + value
        changes.append("added_https_scheme")
    if value.startswith("http://"):
        value = "https://" + value[len("http://"):]
        changes.append("normalized_https")
    mobile = value.replace("://en.m.wikipedia.org", "://en.wikipedia.org")
    if mobile != value:
        value = mobile
        changes.append("normalized_mobile_host")
    without_fragment = value.split("#", 1)[0]
    if without_fragment != value:
        value = without_fragment
        changes.append("removed_fragment")
    without_query = value.split("?", 1)[0]
    if without_query != value:
        value = without_query
        changes.append("removed_query")
    value = value.rstrip("/")
    return value, changes


def split_serialized_links(value: str) -> list[str]:
    serialized = ast.literal_eval(value) if value.startswith("[") else []
    links = []
    pattern = r",\s*(?=(?:https?://|en\.wikipedia\.org/))"
    for item in serialized:
        links.extend(part.strip() for part in re.split(pattern, item) if part.strip())
    return links


def load_questions(path) -> list[dict]:
    """官方 test.tsv -> 每题的 _id / 问题 / 答案 / 去重保序的 canonical wiki URL 列表。

    `question` 与 `answer` 保留 TSV 原文(不 strip),与 pilot/k96_build_queryset.py 当年
    喂给 runner 的那份逐字一致;`question_norm` 才是算指纹用的归一化形式。
    """
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            canonical_urls = []
            seen = set()
            for raw_url in split_serialized_links(row["wiki_links"]):
                canonical_url, _changes = canonicalize_url(raw_url)
                if canonical_url not in seen:
                    canonical_urls.append(canonical_url)
                    seen.add(canonical_url)
            rows.append(
                {
                    "_id": f"frames_{row['']}",
                    "question": row["Prompt"],
                    "answer": row["Answer"],
                    "question_norm": normalize_question(row["Prompt"].strip()),
                    "canonical_urls": canonical_urls,
                }
            )
    return rows


def load_mapping(mapping_path, docs_dir) -> tuple[dict, dict]:
    mapping = json.loads(Path(mapping_path).read_text())
    docs_dir = Path(docs_dir)
    candidates = defaultdict(list)
    mapping_without_member = []
    for key, raw_url in mapping.items():
        path = docs_dir / f"{key}.html"
        canonical_url, _ = canonicalize_url(raw_url)
        if not path.exists():
            mapping_without_member.append({"key": key, "canonical_url": canonical_url})
            continue
        candidates[canonical_url].append(
            {
                "key": key,
                "path": path,
                "raw_url": raw_url,
                "has_fragment": "#" in raw_url,
            }
        )

    selected = {}
    aliases = {}
    for canonical_url, choices in candidates.items():
        ordered = sorted(choices, key=lambda row: (row["has_fragment"], row["key"]))
        selected[canonical_url] = ordered[0]
        aliases[canonical_url] = [row["key"] for row in ordered]
    return selected, {
        "mapping_entries": len(mapping),
        "html_members": len(list(docs_dir.glob("*.html"))),
        "canonical_pages": len(selected),
        "mapping_without_member": len(mapping_without_member),
    }


# ------------------------------------------------------ HTML materializer

def _remove_node(node) -> None:
    parent = node.getparent()
    if parent is not None:
        parent.remove(node)


def _prune_reference_sections(root) -> None:
    children = list(root)
    skip_level = None
    for child in children:
        tag = child.tag.lower() if isinstance(child.tag, str) else ""
        match = re.fullmatch(r"h([2-6])", tag)
        if match:
            level = int(match.group(1))
            heading = normalize_text(" ".join(child.itertext())).lower()
            heading = re.sub(r"\[\s*edit\s*\]\s*$", "", heading).strip()
            if heading in REFERENCE_SECTION_NAMES:
                skip_level = level
                _remove_node(child)
                continue
            if skip_level is not None and level <= skip_level:
                skip_level = None
        if skip_level is not None:
            _remove_node(child)


def _clean_element_text(node) -> str:
    text = normalize_text(" ".join(node.itertext()))
    return text.replace("|", "/")


def _nearest_ancestor_tag(node, tag: str):
    for ancestor in node.iterancestors():
        if isinstance(ancestor.tag, str) and ancestor.tag.lower() == tag:
            return ancestor
    return None


def _linearize_table(table) -> tuple[list[str], int]:
    lines = []
    captions = table.xpath("./caption")
    caption = _clean_element_text(captions[0]) if captions else ""
    lines.append(f"[TABLE] {caption}".rstrip())
    row_count = 0
    seen_rows = set()
    for row in table.xpath(".//tr"):
        if _nearest_ancestor_tag(row, "table") is not table:
            continue
        cells = row.xpath("./th|./td")
        values = [_clean_element_text(cell) for cell in cells]
        values = [value for value in values if value]
        if not values:
            continue
        rendered = " | ".join(values)
        if rendered in seen_rows:
            continue
        seen_rows.add(rendered)
        lines.append(f"| {rendered} |")
        row_count += 1
    lines.append("[END TABLE]")
    return lines, row_count


def parse_revision_metadata(raw_html: bytes) -> tuple[str | None, str | None]:
    text = raw_html.decode("utf-8", errors="ignore")
    revision_match = re.search(r'"wgRevisionId":(\d+)', text)
    modified_match = re.search(r'"dateModified":"([^"]+)"', text)
    modified = modified_match.group(1) if modified_match else None
    if modified is None:
        footer_match = re.search(
            r"This page was last edited on\s+"
            r"(\d{1,2} [A-Z][a-z]+ \d{4}), at (\d{1,2}:\d{2})",
            text,
        )
        if footer_match:
            parsed = datetime.strptime(
                " ".join(footer_match.groups()), "%d %B %Y %H:%M"
            )
            modified = parsed.isoformat(timespec="seconds") + "Z"
    return (
        revision_match.group(1) if revision_match else None,
        modified,
    )


def materialize_html(raw_html: bytes) -> tuple[str, dict]:
    """确定性的 HTML→text:保留标题层级、列表、表格的行列边界,丢掉参考文献段与导航噪声。"""
    from lxml import html

    parser = html.HTMLParser(encoding="utf-8", recover=True)
    document = html.document_fromstring(raw_html, parser=parser)
    roots = document.xpath(
        '//div[contains(concat(" ", normalize-space(@class), " "), '
        '" mw-parser-output ")]'
    )
    # 取 roots[0] 是冻结面的既有行为,别"顺手改成取最长的那个"。部分页面有两个
    # mw-parser-output div,第一个是不含正文的空壳,于是这些页面只物化出标题一行
    # (2026-09-04 复现时实测:测试集 388 个源里 163 个 <200 字符)。榜面所有 FRAMES
    # 数字都是在这份正文上跑出来的,改这一行就复现不出冻结指纹。
    root = roots[0] if roots else document
    title_nodes = document.xpath('//h1[@id="firstHeading"]')
    if not title_nodes:
        title_nodes = document.xpath("//title")
    title = _clean_element_text(title_nodes[0]) if title_nodes else ""

    drop_xpath = (
        ".//script|.//style|.//noscript|"
        './/*[contains(concat(" ", normalize-space(@class), " "), " mw-editsection ")]|'
        './/*[contains(concat(" ", normalize-space(@class), " "), " reference ")]|'
        './/*[contains(concat(" ", normalize-space(@class), " "), " reflist ")]|'
        './/*[contains(concat(" ", normalize-space(@class), " "), " metadata ")]|'
        './/*[contains(concat(" ", normalize-space(@class), " "), " ambox ")]|'
        './/*[contains(concat(" ", normalize-space(@class), " "), " toc ")]|'
        './/*[@style and contains(translate(@style, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", '
        '"abcdefghijklmnopqrstuvwxyz"), "display:none")]'
    )
    for node in list(root.xpath(drop_xpath)):
        _remove_node(node)
    _prune_reference_sections(root)

    lines = [f"# {title}"] if title else []
    table_count = 0
    table_rows = 0
    block_tags = {"p", "li", "dt", "dd", "blockquote", "figcaption"}
    heading_tags = {f"h{i}" for i in range(2, 7)}
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        tag = node.tag.lower()
        if tag == "table":
            if _nearest_ancestor_tag(node, "table") is not None:
                continue
            table_lines, rows = _linearize_table(node)
            lines.extend(table_lines)
            table_count += 1
            table_rows += rows
            continue
        if _nearest_ancestor_tag(node, "table") is not None:
            continue
        if tag in heading_tags:
            value = _clean_element_text(node)
            if value:
                level = int(tag[1])
                lines.append(f"{'#' * level} {value}")
            continue
        if tag not in block_tags:
            continue
        if any(
            isinstance(ancestor.tag, str)
            and ancestor.tag.lower() in block_tags
            for ancestor in node.iterancestors()
        ):
            continue
        value = _clean_element_text(node)
        if not value:
            continue
        prefix = "- " if tag == "li" else ""
        lines.append(prefix + value)

    compact = []
    previous = None
    for line in lines:
        line = normalize_text(line)
        if not line or line == previous:
            continue
        compact.append(line)
        previous = line
    text = "\n".join(compact) + "\n"
    revision_id, modified = parse_revision_metadata(raw_html)
    return text, {
        "title": title,
        "characters": len(text),
        "table_count": table_count,
        "table_rows": table_rows,
        "revision_id": revision_id,
        "date_modified": modified,
    }


# ------------------------------------------------------------ fingerprints
# 逐字搬自 pilot/k96_build_queryset.py。

def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def bundle_hash(ids) -> str:
    return hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16]


def question_hash(question: str) -> str:
    """冻结清单存的是归一化问题的 sha256(见 pilot/k96_freeze_split.py:325)。"""
    return hashlib.sha256(normalize_question(question.strip()).encode()).hexdigest()


# ------------------------------------------------------------- resolution

def _first_existing(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def resolve_split_file(split, data_dir=None, explicit=None):
    """冻结清单的解析顺序:显式 > <data_dir>/frames_variant/<split>_split.jsonl >
    <data_dir>/k96_frames_<split>.jsonl > 站内 data/analysis/。"""
    cands = [explicit]
    for dd in (data_dir, os.environ.get("KVA_DATA_DIR"),
               os.path.expanduser("~/.cache/kvsharearena/data")):
        if dd:
            cands.append(os.path.join(dd, "frames_variant", f"{split}_split.jsonl"))
            cands.append(os.path.join(dd, f"k96_frames_{split}.jsonl"))
    found = _first_existing(*cands)
    if not found:
        raise FileNotFoundError(
            f"找不到 FRAMES {split} 的冻结清单;先 `kva data pull --repo <hf dataset id>` "
            f"或用 --split_file 指定 frames_variant/{split}_split.jsonl"
        )
    return found


def load_frozen(split, data_dir=None, explicit=None):
    path = resolve_split_file(split, data_dir, explicit)
    rows = [json.loads(line) for line in open(path) if line.strip()]
    meta = rows[0].get("_meta", {})
    if meta.get("dataset") != "frames" or meta.get("split") != split:
        raise RuntimeError(f"{path}: _meta 与请求的 dataset/split 不符 ({meta})")
    return rows[1:], {"file": path, "sha256": hash_file(path), "meta": meta}


def resolve_snapshot_dir(explicit=None):
    """显式路径与环境变量无条件优先(目录还不存在也算数,下载会建);
    都没给才在默认落点与站内目录里挑一个已经存在的。"""
    chosen = explicit or os.environ.get("KVA_FRAMES_SNAPSHOT")
    if chosen:
        return chosen
    for cand in (DEFAULT_SNAPSHOT_DIR,):
        if os.path.isdir(cand):
            return cand
    return DEFAULT_SNAPSHOT_DIR


def _locate(root, name):
    """artifact 可能平放在 <root>/,也可能在 <root>/doc_html/(站内 K96 的落盘布局)。"""
    return _first_existing(os.path.join(root, name),
                           os.path.join(root, "doc_html", name))


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  downloading {url}", flush=True)
    # 显式 User-Agent:MLCommons 的 CDN 对 urllib 的默认 UA 直接回 403。
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, open(tmp, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------- snapshot

def ensure_questions(path=None, allow_download=True):
    """官方 test.tsv:显式路径 > HF 缓存 / 下载。返回 (path, sha256)。"""
    if path is None:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=QUESTIONS_REPO, filename=QUESTIONS_FILENAME,
                repo_type="dataset", revision=QUESTIONS_REVISION,
            )
        except Exception as exc:                       # 无 hub 包或离线时退回直连
            if not allow_download:
                raise FileNotFoundError(
                    f"没有官方问题集,且 --no_download 禁止下载({exc})"
                ) from exc
            path = os.path.join(DEFAULT_SNAPSHOT_DIR, QUESTIONS_FILENAME)
            if not os.path.exists(path):
                _download(QUESTIONS_HTTP, path)
    got = hash_file(path)
    if got != QUESTIONS_SHA256:
        raise RuntimeError(
            f"官方 FRAMES 问题集 sha256 漂移:{got} != {QUESTIONS_SHA256}({path})"
        )
    return path, got


def ensure_snapshot(snapshot_dir=None, allow_download=True, extract=True):
    """把 checksum 钉死的 MLPerf HTML 快照准备好。返回 provenance dict。"""
    root = resolve_snapshot_dir(snapshot_dir)
    os.makedirs(root, exist_ok=True)
    provenance = {
        "variant": SNAPSHOT_VARIANT,
        "dir": root,
        "base_url": SNAPSHOT_BASE_URL,
        "md5_manifest_url": SNAPSHOT_MD5_MANIFEST,
        "downloaded": [],
    }

    archive = _locate(root, ARCHIVE_NAME)
    if archive is None:
        if not allow_download:
            raise FileNotFoundError(f"{root} 里没有 {ARCHIVE_NAME},且 --no_download 禁止下载")
        archive = _download(f"{SNAPSHOT_BASE_URL}/doc_html/{ARCHIVE_NAME}",
                            os.path.join(root, ARCHIVE_NAME))
        provenance["downloaded"].append(ARCHIVE_NAME)
    mapping = _locate(root, MAPPING_NAME)
    if mapping is None:
        if not allow_download:
            raise FileNotFoundError(f"{root} 里没有 {MAPPING_NAME},且 --no_download 禁止下载")
        mapping = _download(f"{SNAPSHOT_BASE_URL}/doc_html/{MAPPING_NAME}",
                            os.path.join(root, MAPPING_NAME))
        provenance["downloaded"].append(MAPPING_NAME)

    checks = {
        "archive_md5": hash_file(archive, "md5"),
        "archive_sha256": hash_file(archive),
        "mapping_md5": hash_file(mapping, "md5"),
        "mapping_sha256": hash_file(mapping),
    }
    expected = {
        "archive_md5": ARCHIVE_MD5,
        "archive_sha256": ARCHIVE_SHA256,
        "mapping_md5": MAPPING_MD5,
        "mapping_sha256": MAPPING_SHA256,
    }
    if checks != expected:
        raise RuntimeError(f"快照 checksum 不符:got={checks} expected={expected}")
    provenance.update(checks)
    provenance["archive"] = archive
    provenance["mapping"] = mapping

    # 归属声明(NOTICE/LICENSE):有就记,没有就顺手取一份,取不到不阻塞。
    for name in ATTRIBUTION_NAMES:
        local = _locate(root, name)
        if local is None and allow_download:
            try:
                local = _download(f"{SNAPSHOT_BASE_URL}/doc_html/{name}",
                                  os.path.join(root, name))
                provenance["downloaded"].append(name)
            except Exception as exc:
                provenance.setdefault("attribution_warnings", []).append(f"{name}: {exc}")
        if local:
            provenance.setdefault("attribution_files", {})[name] = local

    docs_dir = os.path.join(root, "docs")
    n_html = len(list(Path(docs_dir).glob("*.html"))) if os.path.isdir(docs_dir) else 0
    if n_html == 0 and extract:
        print(f"  extracting {ARCHIVE_NAME} -> {docs_dir}", flush=True)
        os.makedirs(docs_dir, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                name = os.path.basename(member.name)
                if not member.isfile() or not name.endswith(".html"):
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                with open(os.path.join(docs_dir, name), "wb") as out:
                    out.write(src.read())
        n_html = len(list(Path(docs_dir).glob("*.html")))
    provenance["docs_dir"] = docs_dir
    provenance["html_members"] = n_html
    if n_html == 0:
        raise RuntimeError(f"{docs_dir} 里没有 HTML;快照解包失败")
    return provenance


# ------------------------------------------------------------ page cache

class PageStore:
    """canonical URL -> 物化正文。缓存文件名/格式与 K95 的 page_cache 完全一致,
    因此站内已有的 page_cache 目录可以直接复用,不必重解 2515 个 HTML。"""

    def __init__(self, mapping, cache_dir=None):
        self.mapping = mapping
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"reused": 0, "parsed": 0, "missing": 0}
        self._mem = {}

    def get(self, canonical_url):
        if canonical_url in self._mem:
            return self._mem[canonical_url]
        row = self.mapping.get(canonical_url)
        if row is None:
            self.stats["missing"] += 1
            self._mem[canonical_url] = None
            return None
        text = meta = None
        stem = hashlib.sha256(canonical_url.encode()).hexdigest()
        if self.cache_dir:
            text_path = self.cache_dir / f"{stem}.txt"
            meta_path = self.cache_dir / f"{stem}.json"
            if text_path.exists() and meta_path.exists():
                cached = json.loads(meta_path.read_text())
                if (cached.get("canonical_url") == canonical_url
                        and cached.get("mapping_key") == row["key"]):
                    text = text_path.read_text()
                    if cached.get("text_sha256") == hashlib.sha256(
                            text.encode()).hexdigest():
                        meta, self.stats["reused"] = cached, self.stats["reused"] + 1
                    else:
                        text = None
        if text is None:
            raw_html = Path(row["path"]).read_bytes()
            text, page_meta = materialize_html(raw_html)
            meta = {
                "cache_schema": CACHE_SCHEMA,
                "canonical_url": canonical_url,
                "mapping_key": row["key"],
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "raw_html_sha256": hashlib.sha256(raw_html).hexdigest(),
                **page_meta,
            }
            self.stats["parsed"] += 1
            if self.cache_dir:
                (self.cache_dir / f"{stem}.txt").write_text(text)
                (self.cache_dir / f"{stem}.json").write_text(
                    json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n"
                )
        self._mem[canonical_url] = (text, meta)
        return self._mem[canonical_url]


# ------------------------------------------------------------- provenance

def vendor_provenance(repo_root=None):
    """pilot 原脚本还在盘上时核一下它们没漂;不在(外部用户)就只记 pin。"""
    out = {"vendored_from": dict(VENDORED_FROM), "drift": []}
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel, pinned in VENDORED_FROM.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        got = hash_file(path)
        out.setdefault("checked", {})[rel] = got
        if got != pinned:
            out["drift"].append(
                f"{rel}: 现在是 {got},搬运时是 {pinned} —— 物化/指纹配方可能已改,"
                f"请人工比对后再更新本模块的 pin"
            )
    return out


# ----------------------------------------------------------------- rebuild

def rebuild(out_dir, ids=None, split="test", snapshot_dir=None, questions=None,
            data_dir=None, split_file=None, cache_dir=None, verify=True,
            allow_download=True):
    """把冻结清单里的样本还原成可喂 runner 的载荷,并逐条核对载荷指纹。

    verify=True(默认)时,只要有一条指纹对不上就不写 payloads,只写 report 并抛错——
    对不上说明本地重建的正文与榜面用的不是同一份,拿去跑出来的分数不可比。
    """
    try:                                   # 先于下载 173MB 快照报缺依赖
        import lxml  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "FRAMES 重建需要 lxml(HTML→text 物化):pip install \"kvsharearena[frames]\""
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen, frozen_meta = load_frozen(split, data_dir, split_file)
    if ids:
        wanted = set(ids)
        unknown = wanted - {r["_id"] for r in frozen}
        if unknown:
            raise SystemExit(f"这些 _id 不在 {split} 冻结清单里:{sorted(unknown)}")
        frozen = [r for r in frozen if r["_id"] in wanted]

    print(f"[1/4] 冻结清单 {frozen_meta['file']}(sha256 {frozen_meta['sha256'][:16]}…),"
          f"本次重建 {len(frozen)} 条", flush=True)
    q_path, q_sha = ensure_questions(questions, allow_download)
    print(f"[2/4] 官方问题集 {q_path} sha256 OK", flush=True)
    snap = ensure_snapshot(snapshot_dir, allow_download)
    print(f"[3/4] 快照 checksum OK,{snap['html_members']} 个 HTML @ {snap['docs_dir']}",
          flush=True)

    qa = {row["_id"]: row for row in load_questions(q_path)}
    mapping, mapping_audit = load_mapping(snap["mapping"], snap["docs_dir"])
    if cache_dir is None:
        cache_dir = out_dir / "page_cache"
    store = PageStore(mapping, cache_dir)

    payloads, per_sample, n_match = [], [], 0
    for row in frozen:
        sid = row["_id"]
        fp = row["payload_fingerprint"]
        urls = fp["canonical_urls"]
        entry = {"_id": sid, "n_sources": len(urls), "match": False, "problems": []}

        official_q = qa.get(sid)
        if official_q is None:
            entry["problems"].append("官方问题集里没有这个 _id")
            per_sample.append(entry)
            continue
        if question_hash(official_q["question"]) != fp["question_sha256"]:
            entry["problems"].append("question_sha256 不符(问题原文漂移)")
        if official_q["canonical_urls"] != urls:
            entry["problems"].append(
                f"canonical_urls 与冻结不符:官方={official_q['canonical_urls']}"
            )

        sources, got_hashes, revisions, modifieds = [], [], [], []
        for url in urls:
            page = store.get(url)
            if page is None:
                entry["problems"].append(f"快照里没有这个页面:{url}")
                got_hashes.append(None)
                revisions.append(None)
                modifieds.append(None)
                continue
            text, meta = page
            sources.append({"source_id": url, "text": text})
            got_hashes.append(short_hash(text))
            revisions.append(meta.get("revision_id"))
            modifieds.append(meta.get("date_modified"))

        if len(sources) != row["official"]["n_sources"]:
            entry["problems"].append(
                f"source 条数 {len(sources)} != 官方 n_sources {row['official']['n_sources']}"
            )
        if got_hashes != fp["source_hashes"]:
            bad = [i for i, (g, e) in enumerate(zip(got_hashes, fp["source_hashes"]))
                   if g != e]
            entry["problems"].append(
                f"source_hashes 不符,位次 {bad}:重建={got_hashes} 冻结={fp['source_hashes']}"
            )
        if bundle_hash(urls) != fp["bundle_hash"]:
            entry["problems"].append("bundle_hash 不符")
        if revisions != fp.get("source_revision_ids", revisions):
            entry["problems"].append(
                f"source_revision_ids 不符:重建={revisions} 冻结={fp['source_revision_ids']}"
            )
        if modifieds != fp.get("source_modified_dates", modifieds):
            entry["problems"].append("source_modified_dates 不符(快照页面版本不同)")

        entry["match"] = not entry["problems"]
        entry["bundle_hash"] = fp["bundle_hash"]
        per_sample.append(entry)
        if entry["match"]:
            n_match += 1
            payloads.append({
                "_id": sid,
                "question": official_q["question"],
                "reference_answer": official_q["answer"],
                "sources": sources,
                "official": row["official"],
                "canon_ids": urls,
            })

    report = {
        "schema": "kvsharearena.frames.rebuild-report.v1",
        "split": split,
        "n_requested": len(frozen),
        "n_match": n_match,
        "n_mismatch": len(frozen) - n_match,
        "all_fingerprints_match": n_match == len(frozen),
        "frozen_split": frozen_meta,
        "questions": {"repo": QUESTIONS_REPO, "revision": QUESTIONS_REVISION,
                      "file": q_path, "sha256": q_sha},
        "snapshot": {k: v for k, v in snap.items() if k != "attribution_files"},
        "snapshot_attribution": snap.get("attribution_files", {}),
        "mapping_audit": mapping_audit,
        "page_cache": {"dir": str(cache_dir), **store.stats},
        "provenance": vendor_provenance(),
        "samples": per_sample,
    }
    report_path = out_dir / "frames_rebuild_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    if report["n_mismatch"] and verify:
        bad = [e for e in per_sample if not e["match"]][:5]
        raise SystemExit(
            f"[4/4] REBUILD_REJECTED:{report['n_mismatch']}/{len(frozen)} 条载荷指纹与冻结清单"
            f"不一致,没有写出 payloads。逐条原因见 {report_path};前几条:\n  "
            + "\n  ".join(f"{e['_id']}: {'; '.join(e['problems'])}" for e in bad)
        )

    payload_path = out_dir / "frames_payloads.jsonl"
    with open(payload_path, "w") as handle:
        for record in payloads:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.chmod(payload_path, 0o600)     # 含 gold answer,沿用 K95a2/K96 的卫生规则
    report["payloads"] = {"file": str(payload_path), "n": len(payloads),
                          "sha256": hash_file(payload_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print(f"[4/4] REBUILD_OK {split}: {n_match}/{len(frozen)} 条逐 source 指纹与冻结清单一致\n"
          f"  payloads {payload_path}(含 gold answer,权限 0600)\n"
          f"  report   {report_path}", flush=True)
    return report


def write_fingerprint_index(out_path, splits=("calibration", "test"),
                            data_dir=None, split_files=None):
    """把两个冻结清单压成一份发布用的指纹件:id + 指纹 + 快照/问题集的 pin。

    HF 数据页上 `frames_variant/*_split.jsonl` 已经带逐条指纹,这份件补的是**来源与
    checksum**:重建器要从哪儿下、下到的东西该是什么 md5/sha256。
    """
    split_files = split_files or {}
    index = {
        "schema": "kvsharearena.frames.fingerprints.v1",
        "purpose": "rebuild + verify FRAMES payloads locally; see `kva data rebuild-frames`",
        "dataset_variant": SNAPSHOT_VARIANT,
        "questions": {"repo": QUESTIONS_REPO, "revision": QUESTIONS_REVISION,
                      "file": QUESTIONS_FILENAME, "sha256": QUESTIONS_SHA256,
                      "http": QUESTIONS_HTTP},
        "snapshot": {
            "source": "MLPerf E2E-RAG FRAMES artifact, hosted by MLCommons",
            "base_url": SNAPSHOT_BASE_URL,
            "md5_manifest_url": SNAPSHOT_MD5_MANIFEST,
            "files": {
                f"doc_html/{ARCHIVE_NAME}": {"md5": ARCHIVE_MD5, "sha256": ARCHIVE_SHA256},
                f"doc_html/{MAPPING_NAME}": {"md5": MAPPING_MD5, "sha256": MAPPING_SHA256},
            },
            "note": "2026 HTML crawl, not the FRAMES authors' 2024 snapshot; "
                    "every use of this board is labeled as a dataset variant.",
        },
        "fingerprint_algorithm": {
            "source_hash": "sha256(materialized page text)[:16]",
            "bundle_hash": 'sha256("|".join(sorted(canonical_urls)))[:16]',
            "question_sha256": 'sha256(re.sub(r"\\s+"," ", question.strip().lower()))',
            "materializer": "kvsharearena.frames.materialize_html "
                            "(vendored from pilot/k95_frames_mlperf_audit.py)",
        },
        "provenance": vendor_provenance(),
        "splits": {},
    }
    for split in splits:
        rows, meta = load_frozen(split, data_dir, split_files.get(split))
        index["splits"][split] = {
            "split_file": os.path.basename(meta["file"]),
            "split_file_sha256": meta["sha256"],
            "selector_version": meta["meta"].get("selector_version"),
            "n": len(rows),
            "samples": [
                {
                    "_id": r["_id"],
                    "n_sources": r["official"]["n_sources"],
                    "canonical_urls": r["payload_fingerprint"]["canonical_urls"],
                    "source_hashes": r["payload_fingerprint"]["source_hashes"],
                    "bundle_hash": r["payload_fingerprint"]["bundle_hash"],
                    "question_sha256": r["payload_fingerprint"]["question_sha256"],
                    "source_revision_ids": r["payload_fingerprint"]["source_revision_ids"],
                }
                for r in rows
            ],
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    return index
