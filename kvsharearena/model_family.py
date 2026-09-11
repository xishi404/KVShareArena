#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K77 第二模型泛化 —— 模型族无关的 chat-template / RoPE 构件(两个 Llama 坑的唯一实现入口,
SPEC_K77_K78_K79 §1「写死,别现场发明」)。所有 HF 机器一律 import 这里,防漂移/防现场发明。

设计红线:**Qwen3 路径逐字节保持**(K52/K54/K71/K73b 冻结结果必须仍可复现):
  - chat_kwargs:Qwen3 模板含 enable_thinking 分支 → 传 enable_thinking=False(旧行为);
    Llama-3.1 模板无该变量 → 不传(K75c 先例:apply_chat_template 无 enable_thinking)。
  - build_rope_cache_from_model:复用模型自身 rotary_emb 的 (scaled) inv_freq。
    Qwen3(theta 1e6 无 scaling)→ 与旧 build_rope_cache(theta 公式)逐数值等价;
    Llama-3.1(theta 5e5 + rope_scaling llama3 factor 8)→ inv_freq 已含 llama3 分段缩放,
    硬编码标准 RoPE(仅用 theta)会静默平移错(SPEC pit #1)。
"""
import torch


def chat_kwargs(tok):
    """apply_chat_template 的模型族相关 kwargs。
    Qwen3:{"enable_thinking": False}(保冻结行为);Llama-3.1 / 其它无该变量的模板:{}。"""
    tmpl = tok.chat_template or ""
    return {"enable_thinking": False} if "enable_thinking" in tmpl else {}


def template_pieces(tok):
    """模板差分推导 GEN_HEAD(assistant 生成头)与 USER_CLOSE(user 收尾),模型族无关。
    做法:探针 [system=s, user=SENT];add_generation_prompt True/False 差分 → GEN_HEAD;
          nogen 在 user-content 之后的后缀 → USER_CLOSE。
    Qwen3:GEN_HEAD='<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n', USER_CLOSE='<|im_end|>\\n';
    Llama:GEN_HEAD='<|start_header_id|>assistant<|end_header_id|>\\n\\n',        USER_CLOSE='<|eot_id|>'。"""
    ck = chat_kwargs(tok)
    _U = "%%K77_USER_CONTENT_SENTINEL%%"
    probe = [{"role": "system", "content": "s"}, {"role": "user", "content": _U}]
    nogen = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False, **ck)
    gen = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **ck)
    assert gen.startswith(nogen), "chat template 差分失败(gen 非 nogen 前缀)"
    GEN_HEAD = gen[len(nogen):]
    assert nogen.count(_U) == 1, f"user-content sentinel 命中 {nogen.count(_U)} 次,USER_CLOSE 推导不可靠"
    USER_CLOSE = nogen.split(_U, 1)[1]
    assert USER_CLOSE, "USER_CLOSE 推导为空"
    return GEN_HEAD, USER_CLOSE


def build_rope_cache_from_model(model, max_pos, device):
    """SPEC pit #1:复用模型自身 rotary_emb 的 (scaled) inv_freq 构造 fp32 cos/sin 缓存。
    组合律 R(p+d)=R(d)R(p) 依赖:①角度对位置线性(p·ω,任意 ω 成立);②cos/sin 无位置相关缩放
    → assert attention_scaling==1(llama3 与标准 RoPE 均满足;yarn/longrope 会 ≠1,届时另议)。
    返回 (cos_c, sin_c),形状 [max_pos, head_dim](fp32),索引 delta 即 R(delta)。"""
    rot = model.model.rotary_emb
    inv_freq = rot.inv_freq.detach().to(device=device, dtype=torch.float32)  # [head_dim/2]，含 llama3 缩放
    attn_scaling = float(getattr(rot, "attention_scaling", 1.0))
    assert abs(attn_scaling - 1.0) < 1e-6, \
        f"attention_scaling={attn_scaling} != 1 → shift_k 组合律假设被破坏(非 llama3/标准 RoPE?),停车"
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)             # [max_pos, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)      # [max_pos, head_dim]
    return emb.cos(), emb.sin()


def note_eos_strip_ids(tok, model):
    """笔记尾部要剥的 turn-end token 集合(build_*_notes 冻结笔记用)。
    Qwen:保持冻结行为 = 仅 {tok.eos_token_id}(K71/K73b/K78 冻结笔记据此剥,byte-exact 不可变);
    Llama-3.1:剥完整 eos 集(gen_config.eos_token_id=[128001,128008,128009] 含 <|eot_id|> 128009,
              instruct 实际 turn 收尾 token≠tok.eos_token_id 128001,否则笔记尾残留 <|eot_id|> 污染 KV 段)。
    判据与 chat_kwargs 同(模板含 enable_thinking=Qwen 族)。"""
    tmpl = tok.chat_template or ""
    if "enable_thinking" in tmpl:
        return {tok.eos_token_id}
    gce = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gce is None:
        return {tok.eos_token_id}
    return set(gce) if isinstance(gce, (list, tuple)) else {gce}


def output_health(text):
    """K77 输出健康度(替代 Qwen 专属 think_leak,SPEC pit #2「格式垃圾率/空答率」,K75c §5 口径)。
    summary 观察项(如实报);空答另作笔记生成结构硬闸(空笔记破坏 KV 拼接)。
      empty              : strip 后为空(空答);
      ctrl_leak          : 泄漏 chat 控制 token 字面量('<|'…)——skip_special_tokens=True 下不应出现;
      distinct_word_ratio: 去重词数/总词数(**G-33 连贯闸门:复读检测**,K69b 先例;位置/KV 破坏时 dwr 塌);
      repetition         : dwr<0.5 且非空 = 复读退化(K69b/CacheBlend-on-Llama 位置几何坏征兆);
      garbage            : empty or ctrl_leak or repetition(综合格式垃圾)。"""
    s = (text or "").strip()
    empty = 0 if s else 1
    ctrl_leak = 1 if "<|" in (text or "") else 0
    w = s.split()
    dwr = round(len(set(w)) / len(w), 3) if w else 1.0
    repetition = 1 if (w and dwr < 0.5) else 0
    return {"empty": empty, "ctrl_leak": ctrl_leak, "distinct_word_ratio": dwr,
            "repetition": repetition, "garbage": 1 if (empty or ctrl_leak or repetition) else 0}


def rope_report(model):
    """rope 参数快照(SPEC pit #1 n5 闸门要求:theta/scaling dict 打印入 summary)。"""
    cfg = model.config
    rot = model.model.rotary_emb
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return {
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "attention_scaling": float(getattr(rot, "attention_scaling", 1.0)),
        "head_dim": head_dim,
        "inv_freq_head4": rot.inv_freq.detach().float().cpu().tolist()[:4],
        "inv_freq_tail4": rot.inv_freq.detach().float().cpu().tolist()[-4:],
    }
