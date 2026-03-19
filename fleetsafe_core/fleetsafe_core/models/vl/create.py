from typing import Literal

from fleetsafe_core.models.vl.base import VlModel

VlModelName = Literal["qwen", "moondream"]


def create(name: VlModelName) -> VlModel:
    # This uses inline imports to only import what's needed.
    match name:
        case "qwen":
            from fleetsafe_core.models.vl.qwen import QwenVlModel
            return QwenVlModel()
        case "moondream":
            from fleetsafe_core.models.vl.moondream import MoondreamVlModel
            return MoondreamVlModel()
