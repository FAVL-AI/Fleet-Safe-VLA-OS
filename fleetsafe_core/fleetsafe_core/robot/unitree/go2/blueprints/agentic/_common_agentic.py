#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

from fleetsafe_core.agents.skills.navigation import navigation_skill
from fleetsafe_core.agents.skills.person_follow import person_follow_skill
from fleetsafe_core.agents.skills.speak_skill import speak_skill
from fleetsafe_core.agents.web_human_input import web_input
from fleetsafe_core.core.blueprints import autoconnect
from fleetsafe_core.robot.unitree.go2.connection import GO2Connection
from fleetsafe_core.robot.unitree.unitree_skill_container import unitree_skills

_common_agentic = autoconnect(
    navigation_skill(),
    person_follow_skill(camera_info=GO2Connection.camera_info_static),
    unitree_skills(),
    web_input(),
    speak_skill(),
)

__all__ = ["_common_agentic"]
