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
from collections.abc import Callable

import pytest

from fleetsafe_core.core.core import rpc
from fleetsafe_core.core.module import Module
from fleetsafe_core.core.module_coordinator import ModuleCoordinator
from fleetsafe_core.core.transport import LCMTransport
from fleetsafe_core.msgs.sensor_msgs import PointCloud2
from fleetsafe_core.robot.unitree.type.map import Map as Mapper


@pytest.fixture
def fleetsafe_core():
    ret = ModuleCoordinator()
    ret.start()
    try:
        yield ret
    finally:
        ret.stop()


class Consumer:
    testf: Callable[[int], int]

    def __init__(self, counter=None) -> None:
        self.testf = counter
        self._tasks: set[asyncio.Task[None]] = set()
        print("consumer init with", counter)

    async def waitcall(self, n: int):
        async def task() -> None:
            await asyncio.sleep(n)

            print("sleep finished, calling")
            res = await self.testf(n)
            print("res is", res)

        background_task = asyncio.create_task(task())
        self._tasks.add(background_task)
        background_task.add_done_callback(self._tasks.discard)
        return n


class Counter(Module):
    @rpc
    def addten(self, x: int):
        print(f"counter adding to {x}")
        return x + 10


@pytest.mark.tool
def test_basic(fleetsafe_core) -> None:
    counter = fleetsafe_core.deploy(Counter)
    consumer = fleetsafe_core.deploy(
        Consumer,
        counter=lambda x: counter.addten(x).result(),
    )

    print(consumer)
    print(counter)
    print("starting consumer")
    consumer.start().result()

    res = consumer.inc(10).result()

    print("result is", res)
    assert res == 20


@pytest.mark.tool
def test_mapper_start(fleetsafe_core) -> None:
    mapper = fleetsafe_core.deploy(Mapper)
    mapper.lidar.transport = LCMTransport("/lidar", PointCloud2)
    print("start res", mapper.start().result())


@pytest.mark.tool
def test_counter(fleetsafe_core) -> None:
    counter = fleetsafe_core.deploy(Counter)
    assert counter.addten(10) == 20
