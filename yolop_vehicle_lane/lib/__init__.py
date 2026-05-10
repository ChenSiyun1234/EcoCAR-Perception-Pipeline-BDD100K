"""Lightweight package exports for the YOLOP vehicle+lane project.

Do not eagerly import models, datasets, or training losses here. In Colab/Drive
notebooks, importing ``lib.config`` first should not require every model/loss
file to be present or already refreshed on the mounted filesystem.
"""

from .config import cfg, update_config

__all__ = ["cfg", "update_config"]
