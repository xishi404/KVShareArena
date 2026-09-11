"""
KVShareArena 指标模块 —— 把踩出来的决策固化成可复用、有文档的函数(见 RESEARCH 决策日志)。
质量:PGR(vs floor/oracle)或 retention(vs oracle);算力:reuse_rate/prefill-FLOPs(非墙钟)。
"""
EPS = 1e-3


def pgr(method_q, floor_q, oracle_q):
    """Performance Gap Recovered = (method-floor)/(oracle-floor)。
    floor=consumer-only,oracle=full-recompute(consumer 自己满 prefill 共享上下文)。
    **允许 >1**(跨模型共享超过自算=真实增强);`oracle-floor <= EPS` → None(anchor inversion/collapse,PGR 无定义)。"""
    denom = oracle_q - floor_q
    if denom <= EPS:
        return None
    return (method_q - floor_q) / denom


def retention_vs_oracle(method_q, oracle_q):
    """质量保留 = method/oracle(跨赛道可比;>1=增强,≤1=有损)。用于 KVCOMM 复用侧(reuse/dense)。"""
    return method_q / oracle_q if oracle_q else None


def compute_saved_reuse(reuse_rate):
    """复用侧算力节省(后端无关,主轴)= reuse_rate(省掉的 prefill KV 占比,正)。"""
    return reuse_rate


def compute_saved_flops(method_prefill_flops, oracle_prefill_flops):
    """通用算力节省 = 1 - method/oracle 的 prefill FLOPs。
    C2C 双 prefill(receiver+sharer)> oracle(仅 receiver)→ 负(更贵)。**不用墙钟**(被 decode 主导)。"""
    if not oracle_prefill_flops:
        return None
    return 1.0 - method_prefill_flops / oracle_prefill_flops


def prefill_flops_ratio(receiver_params_b, sharer_params_b=0.0):
    """prefill FLOPs 的参数量代理:receiver(+sharer)相对 receiver-only oracle 的倍数。
    C2C: (recv+sharer)/recv;复用: recv/recv=1(方法侧另用 reuse_rate)。"""
    return (receiver_params_b + sharer_params_b) / receiver_params_b if receiver_params_b else None


def share_score(quality, compute_saved, beta=0.1):
    """RouterArena 式加权调和平均(其 §5.1 原式,线性 β):S_β = (1+β)·q·c/(β·q+c)。
    q=PGR/retention(clip 到 [0,1]),c=compute_saved(clip 到 [0,1])。β 越小越偏质量;
    RouterArena 默认 β=0.1,敏感性须同报 β=1.0(2026-07-20 D-3 定)。
    与原文差异注记:其 cost 为跨数量级美元价格、先 log2 变换;我们 c∈[0,1] 比例,无需变换。
    (2026-07-20 修正:旧版误用 F-beta 的 β² 形式并声称 RouterArena 式——外部审读指认、原文核实后改正。)"""
    q = max(0.0, min(1.0, quality))
    c = max(0.0, min(1.0, compute_saved))
    d = beta * q + c
    return (1.0 + beta) * q * c / d if d > 0 else 0.0


def dominates(a, b):
    """Pareto 支配:a 在(quality, compute_saved)两轴都 ≥ b 且至少一轴 >。"""
    return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])


def pareto_frontier(points):
    """points: [(quality, compute_saved, label)]。返回未被支配的点。"""
    front = []
    for p in points:
        if not any(dominates((q, c), (p[0], p[1])) for q, c, _ in points if (q, c) != (p[0], p[1])):
            front.append(p)
    return front
