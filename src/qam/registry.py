"""Name -> implementation lookup for every pluggable component.

Components register themselves with a decorator and YAML refers to them by name:

    @register("loss", "hybrid_hinge")
    class HybridHingeLoss: ...

    loss_spec:
      func: hybrid_hinge
      kwargs:
        margin: 0.1
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

KINDS = (
    "source",
    "feature",
    "filter",
    "indicator",
    "label",
    "transform",
    "model",
    "loss",
    "splitter",
    "metric",
    "backtest",
)

_REGISTRY: dict[str, dict[str, Any]] = {kind: {} for kind in KINDS}
_loaded = False


def register(kind: str, name: str) -> Callable[[Any], Any]:
    if kind not in _REGISTRY:
        raise ValueError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def decorator(obj: Any) -> Any:
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not obj:
            raise ValueError(f"{kind} {name!r} is already registered by {existing!r}")
        _REGISTRY[kind][name] = obj
        return obj

    return decorator


def load_components() -> None:
    """Import every module under qam.components so their decorators run.

    Submodules and subpackages are discovered automatically, so splitting
    components/models.py into components/models/ needs no change here.
    """
    global _loaded
    if _loaded:
        return
    import qam.components as package

    for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
        importlib.import_module(info.name)
    _loaded = True


def get(kind: str, name: str) -> Any:
    load_components()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "none"
        raise KeyError(f"no {kind} named {name!r}; registered: {known}") from None


def names(kind: str) -> list[str]:
    load_components()
    return sorted(_REGISTRY[kind])


def build(kind: str, spec: Any, **extra: Any) -> Any:
    """Instantiate or call a config Component: get(kind, func)(**kwargs, **extra)."""
    obj = get(kind, spec.func)
    return obj(**{**spec.kwargs, **extra})
