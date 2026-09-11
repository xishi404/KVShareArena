"""HF(transformers 4.52.4)受体后端:冻结考卷的 prompt 骨架、RoPE 位置平移、cache 拼接与贪心解码。

数学与调用序列逐字镜像 pilot/pilot_naive_reuse.py(K54 主榜参照行的生成路径),是所有 HF 后端 driver
共用的受体表面;方法只决定「载荷 KV 怎么出生、怎么修」,受体表面对所有方法恒定(单一变量原则)。
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.cache_utils import DynamicCache

from .model_family import (build_rope_cache_from_model, rope_report, output_health,  # noqa: F401
                           template_pieces, chat_kwargs)

ROPE_MAX_POS = 40960


# ---------- RoPE 重旋转:把已旋转的 K 平移 delta 个位置 ----------
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def shift_k(k, delta, cos_c, sin_c):
    """k: [B,H,T,D](已按位置 p 旋转)→ 旋到 p+delta。R(p+d)=R(d)R(p),对每 T 同一 delta。fp32 计算。"""
    if delta == 0:
        return k
    cos, sin = cos_c[abs(delta)], sin_c[abs(delta)]
    if delta < 0:
        sin = -sin
    kf = k.float()
    return (kf * cos + rotate_half(kf) * sin).to(k.dtype)


# ---------- cache 工具(tf 4.52.4 列表 API) ----------
def cache_slice(cache, s, e):
    """取 [s:e) token 段,返回 legacy 元组列表。"""
    return [(cache.key_cache[i][:, :, s:e, :], cache.value_cache[i][:, :, s:e, :])
            for i in range(len(cache.key_cache))]


def cache_concat(segs, shifts=None, cos_c=None, sin_c=None):
    """segs: list of legacy-list;shifts: 每段 K 的位置平移量(None=不动)。→ DynamicCache"""
    n_layers = len(segs[0])
    layers = []
    for li in range(n_layers):
        ks, vs = [], []
        for si, seg in enumerate(segs):
            k, v = seg[li]
            if shifts and shifts[si]:
                k = shift_k(k, shifts[si], cos_c, sin_c)
            ks.append(k)
            vs.append(v)
        layers.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))
    return DynamicCache.from_legacy_cache(tuple(layers))


def kv_bytes_of(segs):
    """若干 legacy 段实际占用的 KV 字节数(按张量元素数×元素字节,不含分配器开销)。"""
    total = 0
    for seg in segs:
        for k, v in seg:
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return int(total)


def legacy_rope_cache(model, max_pos, device):
    """K67/K68/K69b 时代(kvpress 压缩行冻结件)的 cos/sin 建表方式,逐字复制自 pilot/pilot_kvpress.py:
    在目标 device 上按 config 的 rope_theta/head_dim 现算 inv_freq。与 build_rope_cache_from_model(复用 HF 在
    CPU 上初始化的 inv_freq 缓冲)在个别条目上差 1 ulp,经 fp32 平移→bf16 会在临界处翻转 token——
    移植这些冻结行的 driver 必须用本函数才能逐位复现(K145 A/B 实翻)。"""
    theta = getattr(model.config, "rope_theta", 1000000.0)
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class _FirstTokenClock(StoppingCriteria):
    """记录首个生成 token 落地的时刻(不改变生成结果:恒返回不停止)。"""

    def __init__(self, t0):
        self.t0 = t0
        self.t_first = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.t_first is None:
            torch.cuda.synchronize()
            self.t_first = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class HFReceiver:
    """受体模型 + 冻结模板件。一个进程内只建一次。"""

    def __init__(self, model_id, dtype=torch.bfloat16, device="cuda"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        self.cos_c, self.sin_c = build_rope_cache_from_model(self.model, ROPE_MAX_POS, self.device)
        self.rope = rope_report(self.model)
        self._phase0_gate()
        self.GEN_HEAD, self.USER_CLOSE = template_pieces(self.tok)
        self.CK = chat_kwargs(self.tok)
        self.n_layers = int(self.model.config.num_hidden_layers)

    def _phase0_gate(self):
        """shift_k(δ=0) 恒等 + RoPE 位置 0=identity + 往返(+d,−d)还原;失败即拒跑。"""
        k0 = torch.randn(1, 4, 3, self.cos_c.shape[-1], device=self.device, dtype=torch.bfloat16)
        assert torch.equal(shift_k(k0, 0, self.cos_c, self.sin_c), k0), "[gate] shift_k(δ=0) 非恒等"
        assert torch.allclose(self.cos_c[0], torch.ones_like(self.cos_c[0])) and \
            torch.allclose(self.sin_c[0], torch.zeros_like(self.sin_c[0])), \
            "[gate] RoPE cos[0]!=1/sin[0]!=0(inv_freq 构造错)"
        assert torch.allclose(shift_k(shift_k(k0, 137, self.cos_c, self.sin_c), -137,
                                      self.cos_c, self.sin_c).float(), k0.float(), atol=3e-2), \
            "[gate] shift_k 往返(+137,-137)未还原"

    def ids_of(self, text, add_special=False):
        return self.tok(text, add_special_tokens=add_special, return_tensors="pt").input_ids.to(self.device)

    def open_user_prefix(self, sys_text, user_head):
        """[chat头+sys+user头+user正文开头],去掉模板给 user 消息补的收尾——保持"未闭合"以便裸拼后续段。"""
        full = self.tok.apply_chat_template([{"role": "system", "content": sys_text},
                                             {"role": "user", "content": user_head}],
                                            tokenize=False, add_generation_prompt=False, **self.CK)
        assert full.endswith(self.USER_CLOSE)
        return full[: -len(self.USER_CLOSE)]

    @torch.no_grad()
    def prefill(self, ids, past=None):
        """跑一段 prefill,返回 cache。past 为 None 时新起;pos 由 attention_mask 长度自动推进。"""
        if past is None:
            out = self.model(input_ids=ids, use_cache=True)
            return out.past_key_values
        past_len = past.key_cache[0].shape[2]
        attn = torch.ones((1, past_len + ids.shape[1]), dtype=torch.long, device=self.device)
        cache_pos = torch.arange(past_len, past_len + ids.shape[1], device=self.device)
        out = self.model(input_ids=ids, past_key_values=past, attention_mask=attn,
                         use_cache=True, cache_position=cache_pos)
        return out.past_key_values

    @torch.no_grad()
    def generate_with(self, past, full_ids, max_new):
        """tf 的 prefix-cache 约定:input_ids 传完整序列(past 覆盖段+新段),generate 内部裁剪到 cache 之后。
        返回 (text, sec, ttft_ms)。"""
        past_len = past.key_cache[0].shape[2]
        assert full_ids.shape[1] > past_len, f"full({full_ids.shape[1]}) 必须 > past({past_len})"
        attn = torch.ones((1, full_ids.shape[1]), dtype=torch.long, device=self.device)
        t0 = time.perf_counter()
        clock = _FirstTokenClock(t0)
        out = self.model.generate(input_ids=full_ids, past_key_values=past, attention_mask=attn,
                                  max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id,
                                  stopping_criteria=StoppingCriteriaList([clock]))
        sec = time.perf_counter() - t0
        ttft_ms = (clock.t_first - t0) * 1000.0 if clock.t_first is not None else None
        return self.tok.decode(out[0][full_ids.shape[1]:], skip_special_tokens=True), sec, ttft_ms
