"""Second-stage crop classifier on a frozen class-agnostic MaskDINO.

Deliberately a plain package: nothing here registers a detectron2 META_ARCHITECTURE,
a config namespace or a dataset. The frozen detector is a plain object, and the
classifier is trained by `train_cropcls.py`'s own loop. See spec_cropcls.md section 6.
"""

from .config import CropClsConfig, load_config

__all__ = ["CropClsConfig", "load_config"]
