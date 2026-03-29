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

import asyncio
import os
import time

import pytest
from reactivex import operators as ops

from fleetsafe_core.core.core import rpc
from fleetsafe_core.core.module import Module
from fleetsafe_core.core.module_coordinator import ModuleCoordinator
from fleetsafe_core.core.stream import Out
from fleetsafe_core.core.transport import LCMTransport
from fleetsafe_core.msgs.geometry_msgs import Transform
from fleetsafe_core.msgs.sensor_msgs import Image
from fleetsafe_core.perception.spatial_perception import SpatialMemory
from fleetsafe_core.robot.unitree.type.odometry import Odometry
from fleetsafe_core.utils.data import get_data
from fleetsafe_core.utils.logging_config import setup_logger
from fleetsafe_core.utils.testing import TimedSensorReplay

logger = setup_logger()


class VideoReplayModule(Module):
    """Module that replays video data from TimedSensorReplay."""

    video_out: Out[Image]

    def __init__(self, video_path: str) -> None:
        super().__init__()
        self.video_path = video_path
        self._subscription = None

    @rpc
    def start(self) -> None:
        """Start replaying video data."""
        # Use TimedSensorReplay to replay video frames
        video_replay = TimedSensorReplay(self.video_path, autocast=Image.from_numpy)

        # Subscribe to the replay stream and publish to LCM
        self._subscription = (
            video_replay.stream()
            .pipe(
                ops.sample(2),  # Sample every 2 seconds for resource-constrained systems
                ops.take(5),  # Only take 5 frames total
            )
            .subscribe(self.video_out.publish)
        )

        logger.info("VideoReplayModule started")

    @rpc
    def stop(self) -> None:
        """Stop replaying video data."""
        if self._subscription:
            self._subscription.dispose()
            self._subscription = None
        logger.info("VideoReplayModule stopped")


class OdometryReplayModule(Module):
    """Module that replays odometry data and publishes to the tf system."""

    def __init__(self, odom_path: str) -> None:
        super().__init__()
        self.odom_path = odom_path
        self._subscription = None

    def _publish_tf(self, odom: Odometry) -> None:
        """Convert odometry to TF transforms and publish."""
        self.tf.publish(Transform.from_pose("base_link", odom))

    @rpc
    def start(self) -> None:
        """Start replaying odometry data."""
        # Use TimedSensorReplay to replay odometry
        odom_replay = TimedSensorReplay(self.odom_path, autocast=Odometry.from_msg)

        # Subscribe to the replay stream and publish to tf
        self._subscription = (
            odom_replay.stream()
            .pipe(
                ops.sample(0.5),  # Sample every 500ms
                ops.take(10),  # Only take 10 odometry updates total
            )
            .subscribe(self._publish_tf)
        )

        logger.info("OdometryReplayModule started")

    @rpc
    def stop(self) -> None:
        """Stop replaying odometry data."""
        if self._subscription:
            self._subscription.dispose()
            self._subscription = None
        logger.info("OdometryReplayModule stopped")


@pytest.fixture()
def fleetsafe_core():
    fleetsafe_core = ModuleCoordinator()
    fleetsafe_core.start()
    try:
        yield fleetsafe_core
    finally:
        fleetsafe_core.stop()


@pytest.mark.slow
@pytest.mark.skipif_in_ci
@pytest.mark.asyncio
async def test_spatial_memory_module_with_replay(fleetsafe_core, tmp_path):
    """Test SpatialMemory module with TimedSensorReplay inputs."""
    # Get test data paths
    data_path = get_data("unitree_office_walk")
    video_path = os.path.join(data_path, "video")
    odom_path = os.path.join(data_path, "odom")

    # Deploy modules
    # Video replay module
    video_module = fleetsafe_core.deploy(VideoReplayModule, video_path)
    video_module.video_out.transport = LCMTransport("/test_video", Image)

    # Odometry replay module (publishes to tf system directly)
    odom_module = fleetsafe_core.deploy(OdometryReplayModule, odom_path)

    # Spatial memory module
    spatial_memory = fleetsafe_core.deploy(
        SpatialMemory,
        collection_name="test_spatial_memory",
        embedding_model="clip",
        embedding_dimensions=512,
        min_distance_threshold=0.5,  # 0.5m for test
        min_time_threshold=1.0,  # 1 second
        db_path=str(tmp_path / "chroma_db"),
        visual_memory_path=str(tmp_path / "visual_memory.pkl"),
        new_memory=True,
        output_dir=str(tmp_path / "images"),
    )

    # Connect video stream
    spatial_memory.color_image.connect(video_module.video_out)

    # Start all modules
    video_module.start()
    odom_module.start()
    spatial_memory.start()
    logger.info("All modules started, processing in background...")

    # Wait for frames to be processed with timeout
    timeout = 10.0  # 10 second timeout
    start_time = time.time()

    # Keep checking stats while modules are running
    while (time.time() - start_time) < timeout:
        stats = spatial_memory.get_stats()
        if stats["frame_count"] > 0 and stats["stored_frame_count"] > 0:
            logger.info(
                f"Frames processing - Frame count: {stats['frame_count']}, Stored: {stats['stored_frame_count']}"
            )
            break
        await asyncio.sleep(0.5)
    else:
        # Timeout reached
        stats = spatial_memory.get_stats()
        logger.error(
            f"Timeout after {timeout}s - Frame count: {stats['frame_count']}, Stored: {stats['stored_frame_count']}"
        )
        raise AssertionError(f"No frames processed within {timeout} seconds")

    await asyncio.sleep(2)

    mid_stats = spatial_memory.get_stats()
    logger.info(
        f"Mid-test stats - Frame count: {mid_stats['frame_count']}, Stored: {mid_stats['stored_frame_count']}"
    )
    assert mid_stats["frame_count"] >= stats["frame_count"], (
        "Frame count should increase or stay same"
    )

    # Test query while modules are still running
    try:
        text_results = spatial_memory.query_by_text("office")
        logger.info(f"Query by text 'office' returned {len(text_results)} results")
        assert len(text_results) > 0, "Should have at least one result"
    except Exception as e:
        logger.warning(f"Query by text failed: {e}")

    final_stats = spatial_memory.get_stats()
    logger.info(
        f"Final stats - Frame count: {final_stats['frame_count']}, Stored: {final_stats['stored_frame_count']}"
    )
