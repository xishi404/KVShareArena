"""BaseRepairMethod —— KVShareArena 开放平台的受体侧方法契约(D-43,对应 RouterArena 的 BaseRouter)。

一个方法 = 子类化 BaseRepairMethod + @register(name)。harness 的调用顺序固定为:
    setup(model_id, config)                     # 载入模型/引擎一次
    caches = [build_cache(src) for src in sources]   # 逐源独立编码;harness 保证此时不传 question
    answer(question, caches, config) -> AnswerResult # 复用这些 cache 作答,并回报成本遥测
质量分(F1/PGR)一律由 harness/判分器从 AnswerResult.text 复算,方法不自报质量。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnswerResult:
    """一次作答的产出。成本字段为自报遥测(榜面标 self-reported,维护者抽查)。"""
    text: str
    recomputed_layer_tokens: int = 0        # 作答时对载荷重算的 token×层 数(0=零重算)
    dense_payload_layer_tokens: int = 0     # 载荷满算基准的 token×层 数(算力轴分母)
    kv_bytes: int = 0                       # 作答时实际持有的载荷 KV 字节数(张量上量)
    ttft_ms: Optional[float] = None         # 首 token 时延(同后端内相对 dense prefill 用)
    aux: dict = field(default_factory=dict)


class BaseRepairMethod(ABC):
    """受体侧修复方法。类属性声明适用赛道与机制类别;实例方法实现三步契约。"""
    name: str = "base"
    track: set = frozenset({"re"})          # "re"=Retrieved Evidence(拼接列)/"ar"=Agent Reports(笔记列)
    mechanism_class: int = 0                # 1–9,见排行榜机制类别表;0=未声明
    provenance: str = "port"                # "official" | "port" | "reference"

    @abstractmethod
    def setup(self, model_id: str, config: dict) -> None:
        """按受体模型载入一次。config["exam"] 携带冻结考卷骨架(system prompt/格式段/max_new_tokens)。"""

    @abstractmethod
    def build_cache(self, source_text: str) -> object:
        """独立编码一个来源(question-blind)。返回值只被本方法的 answer 消费,harness 不解释。"""

    @abstractmethod
    def answer(self, question: str, caches: list, config: dict) -> AnswerResult:
        """复用 caches 回答 question。"""

    def teardown(self) -> None:
        pass


_REGISTRY: dict = {}


def register(name: str):
    def deco(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"method name already registered: {name}")
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def get_method(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_methods():
    return sorted(_REGISTRY)
