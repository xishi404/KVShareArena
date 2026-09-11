"""snapkv_r025 —— kvpress 官方 SnapKV(压缩率 0.25)× 拼接错位复用(榜面机制类别 9)。

每个来源孤立编码(位置 0 起,question-blind);**出生时**用 kvpress 官方 SnapKVPress(NVIDIA
forward-hook,库内数学一行不碰)按 head 逐层 topk 物理驱逐 25% 的 KV。作答时保留 key 携出生位旋转,
整段平移到「未压缩逐 token 位」(与免费位置修复 naive_pos 同款绝对位,被驱逐处留洞),query 的
position_ids 落**原始未压缩逻辑长** a_len+ΣL —— 即 kvpress pipeline.py generate_answer 的 vanilla
消费语义(pilot 里记为位置约定 C):key 位 ≤ query 位、无「未来 key」,与 naive_pos 的唯一差别是
被驱逐的 KV 缺席。

逐字镜像 pilot/pilot_kvpress.py 的 `--press snapkv --posconv C --ratio 0.25` 路径(K69b 主榜行)。
解码同样镜像该脚本的手写贪心解码:显式传 position_ids(压缩后 cache 的物理长 ≠ 逻辑位,generate 的
默认位置推断会落错位),cache_position 走物理默认。
"""
import time

import torch

from ..contract import AnswerResult, BaseRepairMethod, register
from ..hf_backend import HFReceiver, cache_concat, cache_slice, kv_bytes_of, legacy_rope_cache

RATIO = 0.25
MIN_COMPRESS_LEN = 64          # = SnapKV 观测窗 window_size:更短的 chunk 统一旁路不压缩


@register("snapkv_r025")
class SnapKVR025(BaseRepairMethod):
    track = frozenset({"re"})
    mechanism_class = 9
    provenance = "official"     # 压缩数学=kvpress 官方库原样调用,本文件只写调用胶水
    backend = "hf"
    training_data = None
    cost_notes = ("recomputed_layer_tokens=0 (no payload recompute; eviction happens at birth); "
                  "dense_payload_layer_tokens = uncompressed payload tokens × layers; "
                  "kv_bytes = bf16 K/V tensor bytes actually held after eviction (≈ (1-ratio) of dense); "
                  "ttft_ms = host time from the prefill call to the first sampled token (cuda-synchronized)")

    def setup(self, model_id, config):
        self.R = HFReceiver(model_id)
        assert hasattr(self.R.model.model, "rotary_emb"), "model.model.rotary_emb 缺失(kvpress hook 依赖)"
        # K69b 冻结行的 RoPE 表(GPU 现算,60000 位);与 HFReceiver 的表差 ulp 级,逐位复现必须用这套
        self.cos_c, self.sin_c = legacy_rope_cache(self.R.model, 60000, self.R.device)
        from kvpress import SnapKVPress                  # 镜像 pilot_kvpress:模型加载之后再导入 kvpress
        self.ratio = float(config.get("ratio", RATIO))
        self.min_compress_len = int(config.get("min_compress_len", MIN_COMPRESS_LEN))
        assert 0.0 <= self.ratio < 1.0, "ratio must be in [0,1)"
        self.press = SnapKVPress(compression_ratio=self.ratio)   # window_size/kernel_size 留官方默认
        eos = self.R.model.generation_config.eos_token_id
        self.eos_ids = eos if isinstance(eos, list) else [eos]

    @torch.no_grad()
    def build_cache(self, source_text):
        R = self.R
        ids = R.ids_of(source_text)
        L = int(ids.shape[1])
        compress = L > self.min_compress_len          # tiny chunk 旁路(压缩观测窗只看 chunk 自身,question-blind)
        if compress:
            with self.press(R.model):
                c = R.model(input_ids=ids, use_cache=True).past_key_values
        else:
            c = R.prefill(ids)
        kept = int(c.key_cache[0].shape[2])
        assert all(c.key_cache[li].shape[2] == kept for li in range(len(c.key_cache))), \
            "press 各层保留长度不一致"
        if compress and self.ratio > 0.0:
            assert kept == int(L * (1 - self.ratio)), f"kept {kept} != int(L*(1-ratio)) (L={L})"
        elif not compress:
            assert kept == L, f"旁路 chunk kept {kept} != L {L}"
        return {"kv": cache_slice(c, 0, kept), "n_tokens": L, "kept": kept}

    def answer(self, question, caches, config):
        R, exam = self.R, config["exam"]
        segA_ids = R.ids_of(R.open_user_prefix(exam["system_prompt"], exam["pre_ctx"]))
        cacheA_full = R.prefill(segA_ids)
        a_len = int(segA_ids.shape[1])
        seg_list, shifts, cur = [cache_slice(cacheA_full, 0, a_len)], [0], a_len
        for c in caches:
            seg_list.append(c["kv"])
            shifts.append(cur)                        # 约定 C:保留 key 落未压缩逐 token 位
            cur += c["n_tokens"]                      # 平移量按**满长**累加(不是 kept)
        cache = cache_concat(seg_list, shifts, self.cos_c, self.sin_c)
        cache_len = int(cache.key_cache[0].shape[2])  # 物理压缩长 = a_len + Σkept
        L_sum = sum(c["n_tokens"] for c in caches)
        kept_sum = sum(c["kept"] for c in caches)
        assert cache_len == a_len + kept_sum, f"cache_len {cache_len} != a_len {a_len} + kept {kept_sum}"
        logical_len = a_len + L_sum                   # 约定 C:query 落原始逻辑长
        segB_ids = R.ids_of(exam["mid_q"] + question + exam["tail"] + R.USER_CLOSE + R.GEN_HEAD)
        text, sec, ttft_ms = self._greedy_decode(cache, segB_ids, logical_len, exam["max_new_tokens"])
        kvb = kv_bytes_of([c["kv"] for c in caches])
        del cache, cacheA_full
        torch.cuda.empty_cache()
        return AnswerResult(text=text, recomputed_layer_tokens=0,
                            dense_payload_layer_tokens=L_sum * R.n_layers,
                            kv_bytes=kvb, ttft_ms=ttft_ms,
                            aux={"sec": sec, "cache_len": cache_len, "a_len": a_len,
                                 "logical_len": logical_len, "query_start": logical_len,
                                 "payload_tokens": int(L_sum), "payload_kept_tokens": int(kept_sum),
                                 "retained_fraction": (kept_sum / L_sum) if L_sum else 1.0,
                                 "n_chunks_compressed": sum(1 for c in caches if c["kept"] < c["n_tokens"])})

    @torch.no_grad()
    def _greedy_decode(self, cache, segB_ids, query_start, max_new):
        """手写贪心解码,显式 position_ids 自 query_start 起(cache_position 走物理默认)。
        与 kvpress generate_answer 同构;do_sample=False ⇒ 纯 argmax。"""
        R = self.R
        pos = torch.arange(query_start, query_start + segB_ids.shape[1], device=R.device).unsqueeze(0)
        t0 = time.perf_counter()
        out = R.model(input_ids=segB_ids, past_key_values=cache, position_ids=pos, use_cache=True)
        cache = out.past_key_values
        nid = out.logits[0, -1].argmax()
        first = nid.item()
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        gen = [nid]
        next_pos = pos[:, -1:] + 1
        if first not in self.eos_ids:
            for i in range(max_new - 1):
                out = R.model(input_ids=gen[-1].view(1, 1), past_key_values=cache,
                              position_ids=next_pos + i, use_cache=True)
                cache = out.past_key_values
                nid = out.logits[0, -1].argmax()
                gen.append(nid)
                if nid.item() in self.eos_ids:
                    break
        return (R.tok.decode(torch.stack(gen), skip_special_tokens=True),
                time.perf_counter() - t0, ttft_ms)

    def teardown(self):
        self.R = None
        self.press = None
        torch.cuda.empty_cache()
