"""
KVShareArena 统一方法接口(对应 RouterArena 的 BaseRouter)。
新方法 = 子类化 BaseSharingMethod + @register,即可进 arena。锚点(oracle/floor/T2T)也是 BaseSharingMethod。

现有 pilot 到本接口的映射(K20 待完整接线,pilot 已验证逻辑):
- C2C(Track A):`prepare` 调 rosetta.utils.evaluate.load_rosetta_model;`run` 用 kv_cache_index 生成,
  reuse_rate=kv_cache_index 中 [1,0] 占比。见 pilot/pilot_c2c_longbench.py。
- KVCOMM(Track C):`prepare` 建 KVCOMM.graph.Graph;`run` 调 graph.arun,
  reuse_rate 从 utils.metrics.metrics_recorder 透传。见 pilot/pilot_kvcomm_*.py。
- 锚点:oracle=full-recompute(consumer 自己满 prefill),floor=consumer-only(无共享上下文),
  T2T=producer 出文本→consumer 读(第二参照线,不进 PGR 分母)。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShareResult:
    """一次方法运行的产出(逐样本)。"""
    output: str
    reuse_rate: float = 0.0                 # 共享 KV 覆盖的 consumer prefill 占比(0=floor,1=全复用)
    ttft_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    prefill_flops: Optional[float] = None   # 后端无关算力主轴(墙钟仅同后端内参考)
    transfer_ms: Optional[float] = None     # 投影/anchor 选择开销
    peak_mem_mb: Optional[float] = None
    input_tokens: int = 0
    gen_tokens: int = 0
    aux: dict = field(default_factory=dict)


class BaseSharingMethod(ABC):
    """每个方法/锚点实现它、调用各自 repo 的既有代码而不重写方法本身。"""
    track: str = "?"        # "A"(跨架构) | "B"(共享基座) | "C"(跨上下文)
    role: str = "method"    # "method" | "oracle" | "floor_consumer_only" | "floor_t2t"
    name: str = "base"

    @abstractmethod
    def prepare(self, producer_model: str, consumer_model: str, config: dict) -> None:
        """按 (producer, consumer, method) 载入一次模型/fuser/adapter/engine。"""

    @abstractmethod
    def run(self, sample: dict) -> ShareResult:
        """跑一条样本(带共享上下文),返回 ShareResult。效率字段能读 repo recorder 就透传。"""

    def teardown(self) -> None:
        pass


_REGISTRY: dict = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def get(name):
    return _REGISTRY[name]


def keys():
    return list(_REGISTRY)
