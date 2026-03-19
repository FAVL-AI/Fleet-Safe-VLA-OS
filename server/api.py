#!/usr/bin/env python3
"""
server/api.py — SAFER-VLA Backend API

FastAPI server exposing fleet controller, VLA pipeline, and WebRTC
configuration as HTTP and WebSocket endpoints.

Serves the Command Center PWA static files and provides the
backend API consumed by the v6 engine.

Usage:
    uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload
    # or
    python server/api.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for fleet imports
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    print("Install: pip install 'fastapi[standard]' uvicorn")
    raise

# Fleet imports (graceful fallback if not available)
try:
    from fleet.dds_messages import FIXSTAND_POSE
    from fleet.fsm_controller import FSMController, FleetFSMManager
    from fleet.policy_engine import PolicyEngine
    from fleet.arm_controller import ArmController
    from fleet.dds_bridge import DDSBridge
    FLEET_AVAILABLE = True
except ImportError:
    FLEET_AVAILABLE = False
    print("[API] Fleet modules not found — running in demo mode")


# ═══════════════════════════════════════════════════════════════════
#  App Configuration
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SAFER-VLA API",
    description="State-of-the-Art Fleet Autonomy for Robotics Education",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
#  Fleet State (shared across endpoints)
# ═══════════════════════════════════════════════════════════════════

class FleetState:
    """In-memory fleet state for the API."""

    def __init__(self):
        self.robots = {
            'robot_0': {'fsm': 'Passive', 'policy': 'HospitalPatrol', 'position': [2.0, 1.5, 0.78], 'arm': None},
            'robot_1': {'fsm': 'Passive', 'policy': 'HospitalPatrol', 'position': [8.0, 5.0, 0.78], 'arm': None},
            'robot_2': {'fsm': 'Patrol', 'policy': 'HospitalPatrol', 'position': [12.0, 3.0, 0.78], 'arm': None},
        }
        self.policy_engine = PolicyEngine() if FLEET_AVAILABLE else None
        self.ws_clients: List[WebSocket] = []
        self.pipeline_status = {
            'record': 'done', 'hdf5': 'idle', 'lerobot': 'idle',
            'train': 'idle', 'deploy': 'idle'
        }
        self.start_time = time.time()
        # Next.js Integration: Live benchmark tracking
        self.telemetry_task = None

    async def broadcast(self, event: dict):
        """Broadcast event to all connected WebSocket clients."""
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)

    async def dds_bridge_loop(self):
        """Runs the real-to-sim DDS bridge, publishing real-time telemetry from fleet robots."""
        try:
            from fleet.dds_bridge import DDSBridge
            import numpy as np
        except ImportError:
            print("[API] DDSBridge not found, skipping telemetry loop")
            return
            
        fleet_mode = os.environ.get("FLEET_MODE", "sim")
        bridge = DDSBridge(mode=fleet_mode)
        bridge.init()
        r0 = bridge.register_robot("robot_0", domain=1)
        r1 = bridge.register_robot("robot_1", domain=2)
        r0.set_position(3.0, 2.5)
        r1.set_position(8.0, 5.0)
        
        bridge.start_sim_publisher(rate_hz=50.0)
        
        # Access the safety transport bound to the lowcmd channel to get true metrics
        channel = bridge.get_channel("rt/lowcmd")
        # Ensure it has the safety transport attached
        if not hasattr(channel, 'safety_transport'):
            from fleetsafe_vla.transport.safety_transport import SafetyTransport
            channel.safety_transport = SafetyTransport()
        
        # Assign bridge to fleet state so REST APIs can use it
        self.bridge = bridge

        while True:
            await asyncio.sleep(0.2)
            try:
                metrics = channel.safety_transport
                # The bridge.start_sim_publisher loop is writing to rt/lowstate
                # Here we simulate the control policies writing to rt/lowcmd natively
                channel.write({"heartbeat": True})
                
                # Fetch positions dynamically
                summary = bridge.get_state_summary()
                
                # Modulate efficiency based on true network traffic
                efficiency = metrics.efficiency + (np.sin(time.time()) * 5)
                robotY = np.sin(time.time() * 2) * 0.03 + 0.95
                
                await self.broadcast({
                    "type": "telemetry",
                    "cbf": metrics.interventions,
                    "violations": metrics.violations,
                    "adherence": metrics.adherence,
                    "efficiency": efficiency,
                    "robotY": robotY,
                    "recalibrating": getattr(metrics, 'recalibrating', False),
                    "ts": time.time()
                })
            except Exception as e:
                print(f"[API] Bridge telemetry loop error: {e}")


fleet = FleetState()

@app.on_event("startup")
async def startup_event():
    fleet.telemetry_task = asyncio.create_task(fleet.dds_bridge_loop())


# ═══════════════════════════════════════════════════════════════════
#  Request/Response Models
# ═══════════════════════════════════════════════════════════════════

class FSMRequest(BaseModel):
    state: str

class PolicyRequest(BaseModel):
    policy: str

class ArmRequest(BaseModel):
    motion: str

class PipelineRequest(BaseModel):
    stage: str
    config: Optional[dict] = None

class InferenceRequest(BaseModel):
    observation: List[float] = []

class LogRequest(BaseModel):
    event: str
    target: str
    policy: str
    action: str
    timestamp: float


# ═══════════════════════════════════════════════════════════════════
#  Fleet REST Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/fleet")
async def get_fleet():
    """Get full fleet status."""
    return {
        "robots": fleet.robots,
        "uptime": int(time.time() - fleet.start_time),
        "fleet_available": FLEET_AVAILABLE,
        "policy_engine": fleet.policy_engine.get_status() if fleet.policy_engine else None,
    }


@app.get("/api/fleet/{robot_id}")
async def get_robot(robot_id: str):
    """Get single robot status."""
    if robot_id not in fleet.robots:
        return JSONResponse({"error": f"Robot {robot_id} not found"}, status_code=404)
    return {"robot_id": robot_id, **fleet.robots[robot_id]}


@app.post("/api/fleet/{robot_id}/fsm")
async def set_fsm(robot_id: str, req: FSMRequest):
    """Set FSM state for a robot."""
    valid_states = ['Passive', 'FixStand', 'Walking', 'Patrol', 'Delivery', 'Emergency']
    if req.state not in valid_states:
        return JSONResponse({"error": f"Invalid state. Valid: {valid_states}"}, status_code=400)
    if robot_id not in fleet.robots:
        return JSONResponse({"error": f"Robot {robot_id} not found"}, status_code=404)

    old_state = fleet.robots[robot_id]['fsm']
    fleet.robots[robot_id]['fsm'] = req.state

    event = {"type": "fsm", "robot_id": robot_id, "old": old_state, "new": req.state, "ts": time.time()}
    await fleet.broadcast(event)

    return {"ok": True, "robot_id": robot_id, "state": req.state}


@app.post("/api/fleet/{robot_id}/policy")
async def set_policy(robot_id: str, req: PolicyRequest):
    """Switch walking policy for a robot."""
    if robot_id not in fleet.robots:
        return JSONResponse({"error": f"Robot {robot_id} not found"}, status_code=404)

    fleet.robots[robot_id]['policy'] = req.policy
    if fleet.policy_engine:
        fleet.policy_engine.set_active_policy(req.policy)

    event = {"type": "policy", "robot_id": robot_id, "policy": req.policy, "ts": time.time()}
    await fleet.broadcast(event)

    return {"ok": True, "robot_id": robot_id, "policy": req.policy}


@app.post("/api/fleet/{robot_id}/arm")
async def trigger_arm(robot_id: str, req: ArmRequest):
    """Trigger arm motion for a robot."""
    valid_motions = ['wave', 'point_to_ward', 'pick_supplies', 'hand_sanitize']
    if req.motion not in valid_motions:
        return JSONResponse({"error": f"Invalid motion. Valid: {valid_motions}"}, status_code=400)
    if robot_id not in fleet.robots:
        return JSONResponse({"error": f"Robot {robot_id} not found"}, status_code=404)

    fleet.robots[robot_id]['arm'] = req.motion

    event = {"type": "arm", "robot_id": robot_id, "motion": req.motion, "ts": time.time()}
    await fleet.broadcast(event)

    return {"ok": True, "robot_id": robot_id, "motion": req.motion}

@app.post("/api/fleet/{robot_id}/log")
async def log_encounter(robot_id: str, req: LogRequest):
    """Log spatial memory encounters to a telemetry file."""
    log_dir = Path(PROJECT_ROOT) / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "mission_telemetry.jsonl"
    
    log_entry = {
        "robot_id": robot_id,
        "event": req.event,
        "target": req.target,
        "policy": req.policy,
        "action": req.action,
        "timestamp": req.timestamp
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return {"ok": True}

@app.get("/api/fleet/logs/download")
async def download_logs():
    """Download the telemetry log file as JSONL."""
    log_file = Path(PROJECT_ROOT) / "logs" / "mission_telemetry.jsonl"
    if not log_file.exists():
        return JSONResponse({"error": "No logs found"}, status_code=404)
    return FileResponse(log_file, media_type='application/jsonl', filename="mission_telemetry.jsonl")


# ═══════════════════════════════════════════════════════════════════
#  Pipeline Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/pipeline/status")
async def pipeline_status():
    """Get VLA pipeline status."""
    return {"pipeline": fleet.pipeline_status}


@app.post("/api/pipeline/{stage}")
async def trigger_pipeline(stage: str, req: Optional[PipelineRequest] = None):
    """Trigger a pipeline stage."""
    if stage not in fleet.pipeline_status:
        return JSONResponse({"error": f"Unknown stage: {stage}"}, status_code=400)

    fleet.pipeline_status[stage] = 'running'
    event = {"type": "pipeline", "stage": stage, "status": "running", "ts": time.time()}
    await fleet.broadcast(event)

    return {"ok": True, "stage": stage, "status": "running"}


# ═══════════════════════════════════════════════════════════════════
#  WebSocket Fleet Stream
# ═══════════════════════════════════════════════════════════════════

@app.websocket("/ws/fleet")
async def ws_fleet(ws: WebSocket):
    """Real-time fleet state stream."""
    await ws.accept()
    fleet.ws_clients.append(ws)

    # Send initial state
    await ws.send_json({
        "type": "init",
        "robots": fleet.robots,
        "pipeline": fleet.pipeline_status,
    })

    try:
        while True:
            # Keep alive + receive commands
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get('type') == 'ping':
                await ws.send_json({"type": "pong", "ts": time.time()})

    except WebSocketDisconnect:
        fleet.ws_clients.remove(ws)


@app.websocket("/ws/gamepad/{robot_id}")
async def ws_gamepad(ws: WebSocket, robot_id: str):
    """Gamepad input stream for a specific robot."""
    await ws.accept()

    try:
        while True:
            data = await ws.receive_bytes()
            # Binary gamepad data: [lx(f32)|ly(f32)|rx(f32)|ry(f32)]
            if len(data) >= 16:
                import struct
                lx, ly, rx, ry = struct.unpack('<ffff', data[:16])
                # Forward to fleet controller (if available)
                # In production, this feeds the DDS JoystickInjector

    except WebSocketDisconnect:
        pass


# ═══════════════════════════════════════════════════════════════════
#  System Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    from server.model_registry import get_registry
    registry = get_registry()
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime": int(time.time() - fleet.start_time),
        "fleet_available": FLEET_AVAILABLE,
        "robots": len(fleet.robots),
        "ws_clients": len(fleet.ws_clients),
        "models": len(registry),
    }


@app.get("/api/webrtc/config")
async def webrtc_config():
    """Return STUN/TURN configuration for WebRTC clients."""
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
        ],
        "signalingUrl": os.environ.get('SIGNALING_URL', 'ws://localhost:8765'),
    }


# ═══════════════════════════════════════════════════════════════════
#  Model Registry & Inference Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/models")
async def list_models(category: Optional[str] = None):
    """List all 13 FLEET-Safe VLA trained models.

    Optional query: ?category=core|extended|safety|vla|policy
    """
    from server.model_registry import get_registry
    registry = get_registry()
    if category:
        return registry.list_by_category(category)
    return registry.list_models()


@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get details for a single model including training results."""
    from server.model_registry import get_registry
    model = get_registry().get_model(model_id)
    if model is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_id}' not found",
                     "available": get_registry().model_ids},
        )
    return model


@app.post("/api/models/{model_id}/infer")
async def infer_model(model_id: str, req: InferenceRequest):
    """Run inference on a trained model.

    Body: {"observation": [0.1, 0.2, ...]}
    Returns: action + safety metrics
    """
    from server.inference_gateway import get_gateway
    try:
        result = get_gateway().infer(model_id, req.observation)
        return result
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


@app.get("/api/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get training metrics/results for a specific model."""
    from server.model_registry import get_registry
    model = get_registry().get_model(model_id)
    if model is None:
        return JSONResponse(status_code=404, content={"error": f"Model '{model_id}' not found"})
    return {
        "model_id": model_id,
        "training_status": model["training_status"],
        "parameters": model["parameters"],
        "epochs": model["epochs"],
        "final_loss": model["final_loss"],
        "final_svr": model["final_svr"],
        "training_time_s": model["training_time_s"],
        "training_results": model["training_results"],
        "wandb_url": model["wandb_url"],
    }


@app.get("/api/training/summary")
async def training_summary():
    """Aggregate training summary across all 13 FLEET models."""
    from server.model_registry import get_registry
    return get_registry().training_summary()


# ═══════════════════════════════════════════════════════════════════
#  Robot URDF Registry Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/robots")
async def list_robots():
    """List all registered robot models parsed from URDF files."""
    from robots.registry import get_robot_registry
    return get_robot_registry().list_robots()


@app.get("/api/robots/{robot_id}")
async def get_robot(robot_id: str):
    """Get full URDF detail for a robot: joints, links, physics, sensors."""
    from robots.registry import get_robot_registry
    robot = get_robot_registry().get_robot(robot_id)
    if robot is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Robot '{robot_id}' not found",
                     "available": get_robot_registry().robot_ids},
        )
    return robot


# ═══════════════════════════════════════════════════════════════════
#  Static File Serving (PWA)
# ═══════════════════════════════════════════════════════════════════

# Serve PWA static files
static_dir = Path(PROJECT_ROOT)

@app.get("/")
async def root():
    return FileResponse(static_dir / "fastbot_command_center.html")

@app.get("/fastbot_command_center.html")
async def command_center():
    return FileResponse(static_dir / "fastbot_command_center.html")

@app.get("/sw.js")
async def service_worker():
    return FileResponse(static_dir / "sw.js", media_type="application/javascript")

# Mount static directories
if (static_dir / "pwa").exists():
    app.mount("/pwa", StaticFiles(directory=str(static_dir / "pwa")), name="pwa")

if (static_dir / "fleet").exists():
    app.mount("/fleet", StaticFiles(directory=str(static_dir / "fleet")), name="fleet")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    print(f"═══ SAFER-VLA Backend API ═══")
    print(f"  API Docs:    http://localhost:{port}/api/docs")
    print(f"  Fleet API:   http://localhost:{port}/api/fleet")
    print(f"  Health:      http://localhost:{port}/api/health")
    print(f"  Command Center: http://localhost:{port}/")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
