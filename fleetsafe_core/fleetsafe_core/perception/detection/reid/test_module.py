# Copyright 2025-2026 Fleet-Safe VLA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from fleetsafe_core.core.transport import LCMTransport
from fleetsafe_core.msgs.foxglove_msgs import ImageAnnotations
from fleetsafe_core.perception.detection.reid.embedding_id_system import EmbeddingIDSystem
from fleetsafe_core.perception.detection.reid.module import ReidModule


@pytest.mark.tool
def test_reid_ingress(imageDetections2d) -> None:
    try:
        from fleetsafe_core.models.embedding import TorchReIDModel
    except Exception:
        pytest.skip("TorchReIDModel not available")

    # Create TorchReID-based IDSystem for testing
    reid_model = TorchReIDModel(model_name="osnet_x1_0")
    reid_model.start()
    idsystem = EmbeddingIDSystem(
        model=lambda: reid_model,
        padding=20,
        similarity_threshold=0.75,
    )

    reid_module = ReidModule(idsystem=idsystem, warmup=False)
    print("Processing detections through ReidModule...")
    reid_module.annotations._transport = LCMTransport("/annotations", ImageAnnotations)
    reid_module.ingress(imageDetections2d)
    reid_module._close_module()
    print("✓ ReidModule ingress test completed successfully")
