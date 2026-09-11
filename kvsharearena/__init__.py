"""KVShareArena —— 跨语境 / 跨权重 KV cache 复用的评测 harness 与判分器。
- contract: BaseRepairMethod / AnswerResult / register(D-43 开放平台契约)。
- harness / cli: `kva run|score|validate|data|methods`。
- scoring: LongBench 官方词面 F1 + PGR + K107 配对 bootstrap。
- base / metrics: K20/K22 时代的 BaseSharingMethod 与指标函数(保留)。"""
__version__ = "0.1.0"

from kvsharearena import base, metrics  # noqa: F401
from kvsharearena.base import BaseSharingMethod, ShareResult  # noqa: F401
from kvsharearena.contract import AnswerResult, BaseRepairMethod, get_method, list_methods, register  # noqa: F401
