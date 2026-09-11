"""内置 driver:import 即注册。snapkv 依赖可选包 kvpress,缺失时静默跳过。"""
from . import naive_pos  # noqa: F401

try:
    from . import snapkv  # noqa: F401
except ImportError:
    pass
