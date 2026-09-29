"""nnscope: watch your NumPy neural network learn.

    import nnscope
    scope = nnscope.Scope("my-mlp")
    scope.log(step, model=model, loss=loss)
"""
__version__ = "0.1.1"

from .adapters import LayerView, extract_layers  # noqa: E402
from .scope import Scope  # noqa: E402

__all__ = ["Scope", "extract_layers", "LayerView", "__version__"]
