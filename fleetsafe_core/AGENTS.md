# Fleet-Safe VLA AGENTS.md

## What is FleetSafeCore

The agentic operating system for generalist robotics. `Modules` communicate via typed streams over LCM, ROS2, DDS, or other transports. `Blueprints` compose modules into runnable robot stacks. `Skills` give agents the ability to execute physical on-hardware function like `grab()`, `follow_object()`, or `jump()`

---

## Quick Start

```bash
# Install
uv sync --all-extras --no-extra dds

# List all runnable blueprints
fleetsafe_core list

# --- Go2 quadruped ---
fleetsafe_core --replay run unitree-go2                  # perception + mapping, replay data
fleetsafe_core --replay run unitree-go2 --daemon         # same, backgrounded
fleetsafe_core --replay run unitree-go2-agentic          # + LLM agent (GPT-4o) + skills
fleetsafe_core --replay run unitree-go2-agentic-mcp      # + McpServer + McpClient (MCP tools live)
fleetsafe_core run unitree-go2-agentic --robot-ip 192.168.123.161  # real Go2 hardware

# --- G1 humanoid ---
fleetsafe_core --simulation run unitree-g1-agentic-sim   # G1 in MuJoCo sim + agent + skills
fleetsafe_core run unitree-g1-agentic --robot-ip 192.168.123.161   # real G1 hardware

# --- Inspect & control ---
fleetsafe_core status
fleetsafe_core log              # last 50 lines, human-readable
fleetsafe_core log -f           # follow/tail in real time
fleetsafe_core agent-send "say hello"
fleetsafe_core stop             # graceful SIGTERM → SIGKILL
fleetsafe_core restart          # stop + re-run with same original args
```

### Blueprint quick-reference

| Blueprint | Robot | Hardware | Agent | MCP server | Notes |
|-----------|-------|----------|-------|------------|-------|
| `unitree-go2-agentic-mcp` | Go2 | real | via McpClient | ✓ | **Only blueprint with McpServer live** |
| `unitree-g1-agentic-sim` | G1 | sim | GPT-4o (G1 prompt) | — | Full agentic sim, no real robot needed |
| `xarm-perception-agent` | xArm | real | GPT-4o | — | Manipulation + perception + agent |
| `xarm7-trajectory-sim` | xArm7 | sim | — | — | Trajectory planning sim |
| `arm-teleop-xarm7` | xArm7 | real | — | — | Quest VR teleop |
| `dual-xarm6-planner` | xArm6×2 | real | — | — | Dual-arm motion planner |

Run `fleetsafe_core list` for the full list.

---

## Tools available to you (MCP)

**MCP only works if the blueprint includes `McpServer`.** Currently the only shipped blueprint that does is `unitree-go2-agentic-mcp`. All other agentic blueprints use the in-process `Agent` module and do NOT expose an MCP endpoint.

```bash
# Start the MCP-enabled blueprint first:
fleetsafe_core --replay run unitree-go2-agentic-mcp --daemon

# Then use MCP tools:
fleetsafe_core mcp list-tools                                              # all available skills as JSON
fleetsafe_core mcp call move --arg x=0.5 --arg duration=2.0               # call by key=value args
fleetsafe_core mcp call move --json-args '{"x": 0.5, "duration": 2.0}'    # call by JSON
fleetsafe_core mcp status      # PID, module list, skill list
fleetsafe_core mcp modules     # module → skills mapping

# Send a message to the running agent (works without McpServer too):
fleetsafe_core agent-send "walk forward 2 meters then wave"
```

The MCP server runs at `http://localhost:9990/mcp` (`GlobalConfig.mcp_port`).

### Adding McpServer to a blueprint

Use **both** `McpServer` and `mcp_client()` — do not mix with `agent()`.

```python
from fleetsafe_core.agents.mcp.mcp_client import mcp_client
from fleetsafe_core.agents.mcp.mcp_server import McpServer

unitree_go2_agentic_mcp = autoconnect(
    unitree_go2_spatial,   # robot stack
    McpServer.blueprint(), # HTTP MCP server — exposes all @skill methods on port 9990
    mcp_client(),          # LLM agent — fetches tools from McpServer
    _common_agentic,       # skill containers
)
```

Reference: `fleetsafe_core/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py`

---

## Repo Structure

```
fleetsafe_core/
├── core/                    # Module system, blueprints, workers, transports
│   ├── module.py            # Module base class, In/Out streams, @rpc, @skill
│   ├── blueprints.py        # Blueprint composition (autoconnect)
│   ├── global_config.py     # GlobalConfig (env vars, CLI flags, .env)
│   └── run_registry.py      # Per-run tracking + log paths
├── robot/
│   ├── cli/fleetsafe_core.py         # CLI entry point (typer)
│   ├── all_blueprints.py    # Auto-generated blueprint registry (DO NOT EDIT MANUALLY)
│   ├── unitree/             # Unitree robot implementations (Go2, G1, B1)
│   │   ├── unitree_skill_container.py  # Go2 @skill methods
│   │   ├── go2/             # Go2 blueprints and connection
│   │   └── g1/              # G1 blueprints, connection, sim, skills
│   └── drone/               # Drone implementations (MAVLink + DJI)
│       ├── connection_module.py        # MAVLink connection
│       ├── camera_module.py            # DJI video stream
│       ├── drone_tracking_module.py    # Visual object tracking
│       └── drone_visual_servoing_controller.py  # Visual servoing
├── agents/
│   ├── agent.py             # Agent module (LangGraph-based)
│   ├── system_prompt.py     # Default Go2 system prompt
│   ├── annotation.py        # @skill decorator
│   ├── mcp/                 # McpServer, McpClient, McpAdapter
│   └── skills/              # NavigationSkillContainer, SpeakSkill, etc.
├── navigation/              # Path planning, frontier exploration
├── perception/              # Object detection, tracking, memory
├── visualization/rerun/     # Rerun bridge
├── msgs/                    # Message types (geometry_msgs, sensor_msgs, nav_msgs)
└── utils/                   # Logging, data loading, CLI tools
docs/
├── usage/modules.md         # ← Module system deep dive
├── usage/blueprints.md      # Blueprint composition guide
├── usage/configuration.md   # GlobalConfig + Configurable pattern
├── development/testing.md   # Fast/slow tests, pytest usage
├── development/fleetsafe_core_run.md # CLI usage, adding blueprints
└── agents/                  # Agent system documentation
```

---

## Architecture

### Modules

Autonomous subsystems. Communicate via `In[T]`/`Out[T]` typed streams. Run in isolated worker processes.

```python
from fleetsafe_core.core.module import Module
from fleetsafe_core.core.stream import In, Out
from fleetsafe_core.core.core import rpc
from fleetsafe_core.msgs.sensor_msgs import Image

class MyModule(Module):
    color_image: In[Image]
    processed: Out[Image]

    @rpc
    def start(self) -> None:
        super().start()
        self.color_image.subscribe(self._process)

    def _process(self, img: Image) -> None:
        self.processed.publish(do_something(img))
```

### Blueprints

Compose modules with `autoconnect()`. Streams auto-connect by `(name, type)` matching.

```python
from fleetsafe_core.core.blueprints import autoconnect

my_blueprint = autoconnect(module_a(), module_b(), module_c())
```

To run a blueprint directly from Python:

```python
# build() deploys all modules into forkserver workers and wires streams
# loop() blocks the main thread until stopped (Ctrl-C or SIGTERM)
autoconnect(module_a(), module_b(), module_c()).build().loop()
```

Expose as a module-level variable for `fleetsafe_core run` to find it. Add to the registry by running `pytest fleetsafe_core/robot/test_all_blueprints_generation.py`.

### GlobalConfig

Singleton config. Values cascade: defaults → `.env` → env vars → blueprint → CLI flags. Env vars prefixed `FLEETSAFE_CORE_`. Key fields: `robot_ip`, `simulation`, `replay`, `viewer`, `n_workers`, `mcp_port`.

### Transports

- **LCMTransport**: Default. Multicast UDP.
- **SHMTransport/pSHMTransport**: Shared memory — use for images and point clouds.
- **pLCMTransport**: Pickled LCM — use for complex Python objects.
- **ROSTransport**: ROS topic bridge — interop with ROS nodes (`fleetsafe_core/core/transport.py`).
- **DDSTransport**: DDS pub/sub — available when `DDS_AVAILABLE`; install with `uv sync --extra dds` (`fleetsafe_core/protocol/pubsub/impl/ddspubsub.py`).

---

## CLI Reference

### Global flags

Every `GlobalConfig` field is a CLI flag: `--robot-ip`, `--simulation/--no-simulation`, `--replay/--no-replay`, `--viewer {rerun|rerun-web|foxglove|none}`, `--mcp-port`, `--n-workers`, etc. Flags override `.env` and env vars.

### Core commands

| Command | Description |
|---------|-------------|
| `fleetsafe_core run <blueprint> [--daemon]` | Start a blueprint |
| `fleetsafe_core status` | Show running instance (run ID, PID, blueprint, uptime, log path) |
| `fleetsafe_core stop [--force]` | SIGTERM → SIGKILL after 5s; `--force` = immediate SIGKILL |
| `fleetsafe_core restart [--force]` | Stop + re-exec with original args |
| `fleetsafe_core list` | List all non-demo blueprints |
| `fleetsafe_core show-config` | Print resolved GlobalConfig values |
| `fleetsafe_core log [-f] [-n N] [--json] [-r <run-id>]` | View per-run logs |
| `fleetsafe_core mcp list-tools / call / status / modules` | MCP tools (requires McpServer in blueprint) |
| `fleetsafe_core agent-send "<text>"` | Send text to the running agent via LCM |
| `fleetsafe_core lcmspy / agentspy / humancli / top` | Debug/diagnostic tools |
| `fleetsafe_core topic echo <topic> / send <topic> <expr>` | LCM topic pub/sub |
| `fleetsafe_core rerun-bridge` | Launch Rerun visualization standalone |

Log files: `~/.local/state/fleetsafe_core/logs/<run-id>/main.jsonl`
Run registry: `~/.local/state/fleetsafe_core/runs/<run-id>.json`

---

## Agent System

### The `@skill` Decorator

`fleetsafe_core/agents/annotation.py`. Sets `__rpc__ = True` and `__skill__ = True`.

- `@rpc` alone: callable via RPC, not exposed to LLM
- `@skill`: implies `@rpc` AND exposes method to the LLM as a tool. **Do not stack both.**

#### Schema generation rules

| Rule | What happens if you break it |
|------|------------------------------|
| **Docstring is mandatory** | `ValueError` at startup — module fails to register, all skills disappear |
| **Type-annotate every param** | Missing annotation → no `"type"` in schema — LLM has no type info |
| **Return `str`** | `None` return → agent hears "It has started. You will be updated later." |
| **Full docstring verbatim in `description`** | Keep `Args:` block concise — it appears in every tool-call prompt |

Supported param types: `str`, `int`, `float`, `bool`, `list[str]`, `list[float]`. Avoid complex nested types.

#### Minimal correct skill

```python
from fleetsafe_core.agents.annotation import skill
from fleetsafe_core.core.core import rpc
from fleetsafe_core.core.module import Module

class MySkillContainer(Module):
    @rpc
    def start(self) -> None:
        super().start()

    @rpc
    def stop(self) -> None:
        super().stop()

    @skill
    def move(self, x: float, duration: float = 2.0) -> str:
        """Move the robot forward or backward.

        Args:
            x: Forward velocity in m/s. Positive = forward, negative = backward.
            duration: How long to move in seconds.
        """
        return f"Moving at {x} m/s for {duration}s"

my_skill_container = MySkillContainer.blueprint

### System Prompts

| Robot | File | Variable |
|-------|------|----------|
| Go2 (default) | `fleetsafe_core/agents/system_prompt.py` | `SYSTEM_PROMPT` |
| G1 humanoid | `fleetsafe_core/robot/unitree/g1/system_prompt.py` | `G1_SYSTEM_PROMPT` |

Pass the robot-specific prompt: `agent(system_prompt=G1_SYSTEM_PROMPT)`. Agent defaults to Go2 — wrong prompt causes hallucinated skills.

### RPC Wiring

To call methods on another module, declare a `Spec` Protocol and annotate an attribute with it. The blueprint injects the matching module at build time — fully typed, no strings, fails at build time (not runtime) if no match is found.

```python
# my_module_spec.py
from typing import Protocol
from fleetsafe_core.spec.utils import Spec

class NavigatorSpec(Spec, Protocol):
    def set_goal(self, goal: PoseStamped) -> bool: ...
    def cancel_goal(self) -> bool: ...

# my_skill_container.py
class MySkillContainer(Module):
    _navigator: NavigatorSpec   # injected by blueprint at build time

    @skill
    def go_to(self, x: float, y: float) -> str:
        """Navigate to a position."""
        self._navigator.set_goal(make_pose(x, y))
        return "Navigating"
```

If multiple modules match the spec, use `.remappings()` to resolve. Source: `fleetsafe_core/spec/utils.py`, `fleetsafe_core/core/blueprints.py`.

**Legacy**: existing skill containers use `rpc_calls: list[str]` + `get_rpc_calls("ClassName.method")`. This still works but wiring failures are silent and only surface at runtime. Don't use it in new code.

### Adding a New Skill

1. Pick the right container (robot-specific or `fleetsafe_core/agents/skills/`).
2. `@skill` + mandatory docstring + type annotations on all params.
3. If it needs another module's RPC, use the Spec pattern.
4. Return a descriptive `str`.
5. Update the system prompt — add to the `# AVAILABLE SKILLS` section.
6. Expose as `my_container = MySkillContainer.blueprint` and include in the agentic blueprint.

---

## Testing

```bash
# Fast tests (default)
uv run pytest

# Include slow tests (CI)
./bin/pytest-slow

# Single file
uv run pytest fleetsafe_core/core/test_blueprints.py -v

# Mypy
uv run mypy fleetsafe_core/
```

`uv run pytest` excludes `slow`, `tool`, and `mujoco` markers. CI (`./bin/pytest-slow`) includes slow, excludes tool and mujoco. See `docs/development/testing.md`.

---

## Pre-commit & Code Style

Pre-commit runs on `git commit`. Includes ruff format/check, license headers, LFS checks.

**Always activate the venv before committing:** `source .venv/bin/activate`

Code style rules:
- Imports at top of file. No inline imports unless circular dependency.
- Use `requests` for HTTP (not `urllib`). Use `Any` (not `object`) for JSON values.
- Prefix manual test scripts with `demo_` to exclude from pytest collection.
- Don't hardcode ports/URLs — use `GlobalConfig` constants.
- Type annotations required. Mypy strict mode.

---

## `all_blueprints.py` is auto-generated

`fleetsafe_core/robot/all_blueprints.py` is generated by `test_all_blueprints_generation.py`. After adding or renaming blueprints:

```bash
pytest fleetsafe_core/robot/test_all_blueprints_generation.py
```

CI asserts the file is current — if it's stale, CI fails.

---

## Git Workflow

- Branch prefixes: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `chore/`, `perf/`
- **PRs target `dev`** — never push to `main` or `dev` directly
- **Don't force-push** unless after a rebase with conflicts
- **Minimize pushes** — every push triggers CI (~1 hour on self-hosted runners). Batch commits locally, push once.

---

## Further Reading

- Module system: `docs/usage/modules.md`
- Blueprints: `docs/usage/blueprints.md`
- Visualization: `docs/usage/visualization.md`
- Configuration: `docs/usage/configuration.md`
- Testing: `docs/development/testing.md`
- CLI / fleetsafe_core run: `docs/development/fleetsafe_core_run.md`
- LFS data: `docs/development/large_file_management.md`
- Agent system: `docs/agents/`
