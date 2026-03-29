import time
import random

class RustTransport:
    """
    Mocked Rust DDS Transport interface for FleetSafe OS.
    Eliminates compilation overhead and third-party branding for immediate UI testing.
    """
    def __init__(self):
        self.connected = True
        print(f"[{time.time():.4f}] FleetOS Rust Engine: Native DDS Transport initialized")

    def publish(self, topic: str, msg: dict):
        # Extremely fast mock publication simulating memory-mapped zero-copy
        pass

class FleetEngineTelemetry:
    """
    Mocked metrics for the lower-level Rust control loop.
    Replaces "DiMOS" branding requested in the implementation plan.
    """
    @staticmethod
    def get_metrics():
        class Metrics:
            # Simulate high-speed native loop rates (800Hz-1kHz)
            control_freq = random.randint(850, 1050)
            # Simulate sub-millisecond DDS latency (150-300 microseconds)
            msg_latency_us = random.randint(150, 320)
        return Metrics()
