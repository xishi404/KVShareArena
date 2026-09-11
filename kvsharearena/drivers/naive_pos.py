"""naive_pos —— 免费位置修复(参照行,榜面机制类别 3)。

每个来源孤立编码(位置从 0 起,question-blind),作答时把各段 K 用 RoPE 重旋转平移到目标绝对位置后
直接拼接;不重算任何载荷 token、不修段间注意力。逐字镜像 pilot/pilot_naive_reuse.py 的
`--cond naive_concat_pos` 路径(K54 主榜参照行)。
"""
import torch

from ..contract import AnswerResult, BaseRepairMethod, register
from ..hf_backend import HFReceiver, cache_concat, cache_slice, kv_bytes_of


@register("naive_pos")
class NaivePos(BaseRepairMethod):
    track = frozenset({"re"})
    mechanism_class = 3
    provenance = "reference"
    backend = "hf"
    training_data = None
    cost_notes = ("recomputed_layer_tokens=0 (no payload recompute); dense_payload_layer_tokens = "
                  "payload tokens × layers; kv_bytes = bf16 K/V tensor bytes of all payload segments; "
                  "ttft_ms = host time from generate() call to first token (cuda-synchronized)")

    def setup(self, model_id, config):
        self.R = HFReceiver(model_id)

    def build_cache(self, source_text):
        R = self.R
        ids = R.ids_of(source_text)
        c = R.prefill(ids)                          # 孤立编码:位置 0 起
        return {"ids": ids, "kv": cache_slice(c, 0, ids.shape[1]), "n_tokens": int(ids.shape[1])}

    def answer(self, question, caches, config):
        R, exam = self.R, config["exam"]
        segA_ids = R.ids_of(R.open_user_prefix(exam["system_prompt"], exam["pre_ctx"]))
        cacheA_full = R.prefill(segA_ids)
        a_len = segA_ids.shape[1]
        segA = cache_slice(cacheA_full, 0, a_len)
        seg_list, shifts, id_list, cur = [segA], [0], [segA_ids], a_len
        for c in caches:
            seg_list.append(c["kv"])
            shifts.append(cur)                      # 平移到目标绝对位置
            id_list.append(c["ids"])
            cur += c["n_tokens"]
        cache = cache_concat(seg_list, shifts, R.cos_c, R.sin_c)
        cache_len = cache.key_cache[0].shape[2]
        segB_ids = R.ids_of(exam["mid_q"] + question + exam["tail"] + R.USER_CLOSE + R.GEN_HEAD)
        full_ids = torch.cat(id_list + [segB_ids], dim=1)
        assert full_ids.shape[1] == cache_len + segB_ids.shape[1]
        text, sec, ttft_ms = R.generate_with(cache, full_ids, exam["max_new_tokens"])
        payload_tokens = sum(c["n_tokens"] for c in caches)
        kvb = kv_bytes_of([c["kv"] for c in caches])
        del cache
        torch.cuda.empty_cache()
        return AnswerResult(text=text, recomputed_layer_tokens=0,
                            dense_payload_layer_tokens=payload_tokens * R.n_layers,
                            kv_bytes=kvb, ttft_ms=ttft_ms,
                            aux={"sec": sec, "cache_len": int(cache_len), "a_len": int(a_len),
                                 "payload_tokens": int(payload_tokens)})

    def teardown(self):
        self.R = None
        torch.cuda.empty_cache()
