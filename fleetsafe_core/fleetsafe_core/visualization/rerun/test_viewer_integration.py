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

"""Tests for fleetsafe_core-viewer integration with RerunBridgeModule.

These tests verify that:
1. The fleetsafe_core-viewer binary is installed and discoverable
2. rerun_bindings.spawn() accepts the executable_name parameter
3. bridge.py has the correct spawn logic
4. Version compatibility between rerun-sdk and fleetsafe_core-viewer

These run in CI where fleetsafe_core-viewer is a core dependency, so the binary
is always available. The main risk we're guarding against is rerun-sdk
pushing an update that breaks the spawn interface or version compatibility.
"""

import inspect
import re
import shutil


class TestViewerBinaryInstallation:
    """Verify fleetsafe_core-viewer binary is installed and functional."""

    def test_binary_on_path(self):
        """fleetsafe_core-viewer binary must be discoverable on PATH."""
        path = shutil.which("fleetsafe_core-viewer")
        assert path is not None, (
            "fleetsafe_core-viewer binary not found on PATH. "
            "Ensure 'fleetsafe_core-viewer' is in pyproject.toml dependencies."
        )

    def test_binary_executable(self):
        """fleetsafe_core-viewer binary must be executable."""
        import os

        path = shutil.which("fleetsafe_core-viewer")
        assert path is not None
        assert os.access(path, os.X_OK), f"fleetsafe_core-viewer at {path} is not executable"


class TestRerunBindingsInterface:
    """Verify rerun_bindings.spawn() interface hasn't changed."""

    def test_spawn_accepts_executable_name(self):
        """rerun_bindings.spawn must accept executable_name kwarg.

        This is the mechanism we use to launch fleetsafe_core-viewer instead of
        stock rerun. If rerun-sdk removes this parameter, our integration
        breaks silently (falls back to stock rerun).
        """
        import rerun_bindings

        sig = inspect.signature(rerun_bindings.spawn)
        assert "executable_name" in sig.parameters, (
            "rerun_bindings.spawn() no longer accepts 'executable_name'. "
            "This means rerun-sdk changed its spawn interface. "
            "The fleetsafe_core-viewer integration in bridge.py will fail."
        )

    def test_spawn_accepts_port(self):
        """rerun_bindings.spawn must accept port kwarg."""
        import rerun_bindings

        sig = inspect.signature(rerun_bindings.spawn)
        assert "port" in sig.parameters, "rerun_bindings.spawn() no longer accepts 'port'. "

    def test_spawn_accepts_expected_params(self):
        """All spawn params used by bridge.py must be available."""
        import rerun_bindings

        sig = inspect.signature(rerun_bindings.spawn)
        required = {"port", "executable_name"}
        missing = required - set(sig.parameters.keys())
        assert not missing, (
            f"rerun_bindings.spawn() missing parameters: {missing}. "
            "rerun-sdk may have changed its interface."
        )


class TestBridgeSpawnLogic:
    """Verify bridge.py has the correct fleetsafe_core-viewer spawn logic."""

    def test_bridge_references_fleetsafe_core_viewer(self):
        """bridge.py must attempt to spawn fleetsafe_core-viewer."""
        from fleetsafe_core.visualization.rerun.bridge import RerunBridgeModule

        src = inspect.getsource(RerunBridgeModule.start)
        assert "fleetsafe_core-viewer" in src, (
            "bridge.py start() does not reference 'fleetsafe_core-viewer'. "
            "The viewer integration may have been removed."
        )

    def test_bridge_uses_rerun_bindings(self):
        """bridge.py must use rerun_bindings (not subprocess) for spawn."""
        from fleetsafe_core.visualization.rerun.bridge import RerunBridgeModule

        src = inspect.getsource(RerunBridgeModule.start)
        assert "rerun_bindings" in src, "bridge.py start() does not use rerun_bindings. "

    def test_bridge_has_fallback(self):
        """bridge.py must fall back to stock rerun if fleetsafe_core-viewer unavailable."""
        from fleetsafe_core.visualization.rerun.bridge import RerunBridgeModule

        src = inspect.getsource(RerunBridgeModule.start)
        assert "ImportError" in src or "except" in src, (
            "bridge.py start() has no fallback for missing fleetsafe_core-viewer. "
            "Users without fleetsafe_core-viewer will crash."
        )


def _parse_version(version_str: str) -> tuple[int, int]:
    """Extract (major, minor) from a version string like '0.29.2' or '0.30.0a2'."""
    match = re.match(r"(\d+)\.(\d+)", version_str)
    assert match, f"Cannot parse version: {version_str}"
    return int(match.group(1)), int(match.group(2))


class TestVersionCompatibility:
    """Catch rerun-sdk / fleetsafe_core-viewer version drift before it bites us at runtime."""

    def test_versions_within_one_minor(self):
        """rerun-sdk and fleetsafe_core-viewer must be within 1 minor version.

        fleetsafe_core-viewer is built from a rerun foundation, so they track the same
        release line. If they drift by more than one minor version, the
        gRPC protocol or internal APIs are likely incompatible.
        """
        import importlib.metadata

        import rerun

        sdk_version = rerun.__version__
        viewer_version = importlib.metadata.version("fleetsafe_core-viewer")

        sdk_major, sdk_minor = _parse_version(sdk_version)
        viewer_major, viewer_minor = _parse_version(viewer_version)

        assert sdk_major == viewer_major, (
            f"Major version mismatch: rerun-sdk={sdk_version}, fleetsafe_core-viewer={viewer_version}. "
            f"These are likely incompatible."
        )
        assert abs(sdk_minor - viewer_minor) <= 1, (
            f"Version drift too large: rerun-sdk={sdk_version}, fleetsafe_core-viewer={viewer_version}. "
            f"Update fleetsafe_core-viewer to match rerun-sdk or vice versa."
        )
