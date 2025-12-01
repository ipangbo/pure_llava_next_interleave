"""Lightweight LLaVA-NeXT-Interleave package."""

from importlib import metadata

__all__ = ["__version__"]

try:
    __version__ = metadata.version("llava_next_interleave")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
