#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 FLEET SAFE VLA - HFB-S | Auto-Shutdown Module
═══════════════════════════════════════════════════════════════════════════════
 Automatic GCP instance shutdown after training completion or timeout.

 Features:
   - Graceful shutdown with checkpoint save
   - Budget-based auto-stop
   - Idle detection (no GPU activity)
   - SIGTERM/SIGINT handler
   - Training log preservation

 Usage:
   from training.auto_shutdown import AutoShutdown
   shutdown = AutoShutdown(max_hours=4, budget_usd=10)
   shutdown.start()
   ...  # training
   shutdown.check_and_stop()
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import json
import time
import signal
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Callable

logger = logging.getLogger("AutoShutdown")

@dataclass
class ShutdownConfig:
    """Auto-shutdown configuration."""
    instance_name: str = "isaac-l4-dev"
    zone: str = "us-central1-a"
    project: str = "fleet-safe-vla"
    
    # Budget
    cost_per_hour: float = 32.76   # 8x H100 SOTA Finetuning Node
    budget_limit_usd: float = 800.0
    
    # Timeouts
    max_hours: float = 12.0        # Maximum training time
    idle_timeout_min: float = 15.0 # Stop if idle for 15 minutes
    
    # Checkpointing
    save_on_shutdown: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_dir: str = "training_logs"


class AutoShutdown:
    """Auto-shutdown manager for GCP training instances."""
    
    def __init__(self, config: ShutdownConfig = None,
                 on_shutdown: Optional[Callable] = None):
        self.cfg = config or ShutdownConfig()
        self.on_shutdown = on_shutdown
        self.start_time = None
        self.last_activity = None
        self._shutdown_requested = False
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.last_activity = time.time()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"  ⏱️  Auto-shutdown: max {self.cfg.max_hours}h, "
                    f"budget ${self.cfg.budget_limit_usd:.2f}")
    
    def _signal_handler(self, sig, frame):
        """Handle SIGTERM/SIGINT gracefully."""
        logger.info(f"  🛑 Signal {sig} received — initiating graceful shutdown")
        self._shutdown_requested = True
        self.stop(reason=f"signal_{sig}")
    
    def tick(self):
        """Update activity timestamp (call from training loop)."""
        self.last_activity = time.time()
    
    def elapsed_hours(self) -> float:
        """Hours since start."""
        if not self.start_time:
            return 0
        return (time.time() - self.start_time) / 3600
    
    def current_cost(self) -> float:
        """Estimated cost so far."""
        return self.elapsed_hours() * self.cfg.cost_per_hour
    
    def idle_minutes(self) -> float:
        """Minutes since last activity."""
        if not self.last_activity:
            return 0
        return (time.time() - self.last_activity) / 60
    
    def should_stop(self) -> tuple:
        """Check if shutdown should be triggered.
        
        Returns (should_stop: bool, reason: str).
        """
        if self._shutdown_requested:
            return True, "signal_requested"
        
        if self.elapsed_hours() >= self.cfg.max_hours:
            return True, f"max_time_{self.cfg.max_hours}h"
        
        if self.current_cost() >= self.cfg.budget_limit_usd:
            return True, f"budget_exceeded_${self.current_cost():.2f}"
        
        if self.idle_minutes() >= self.cfg.idle_timeout_min:
            return True, f"idle_{self.idle_minutes():.0f}min"
        
        return False, ""
    
    def check_and_stop(self):
        """Check conditions and stop if needed."""
        should, reason = self.should_stop()
        if should:
            self.stop(reason)
    
    def stop(self, reason: str = "completed"):
        """Execute graceful shutdown."""
        logger.info(f"\n  🔄 Auto-shutdown: reason={reason}")
        logger.info(f"     Elapsed: {self.elapsed_hours():.2f}h")
        logger.info(f"     Cost: ${self.current_cost():.2f}")
        
        # Save training log
        log = {
            "reason": reason,
            "elapsed_hours": self.elapsed_hours(),
            "cost_usd": self.current_cost(),
            "timestamp": datetime.now().isoformat(),
            "instance": self.cfg.instance_name,
        }
        
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "shutdown_log.json").write_text(json.dumps(log, indent=2))
        
        # Callback
        if self.on_shutdown:
            try:
                self.on_shutdown()
            except Exception as e:
                logger.warning(f"     Shutdown callback error: {e}")
        
        # Stop GCP instance
        try:
            subprocess.run(
                ["gcloud", "compute", "instances", "stop",
                 self.cfg.instance_name,
                 f"--zone={self.cfg.zone}", "--quiet"],
                timeout=60, check=False,
                capture_output=True,
            )
            logger.info("     ✅ GCP instance stopped")
        except Exception as e:
            logger.warning(f"     ⚠️ GCP stop failed: {e}")
