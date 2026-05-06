from __future__ import annotations

from importlib import import_module

_tk_backend = import_module("matplotlib.backends.backend_tkagg")

FigureCanvasTkAgg = _tk_backend.FigureCanvasTkAgg

__all__ = ["FigureCanvasTkAgg"]
