"""
spatial_registration.py: Drift-Aware Spatial Registration Guard

This module implements constraints for multi-frame 3D reconstruction alignment,
acting as a strict geometric firewall to prevent catastrophic drift. 
If Classical math attempts to shift a frame by >5% of the scene's bounding box
diagonal relative to the immutable anchor, the hypothesis is instantly rejected.
"""
import numpy as np
import logging
from typing import Optional, Tuple, Dict

class DriftAwareRegistration:
    """
    Constrained optimization firewall for spatial alignments.
    Ensures that frame-to-frame shifts obey strict topological limits.
    """
    def __init__(self, shift_limit_ratio: float = 0.05):
        self.shift_limit_ratio = shift_limit_ratio
        self.anchor_frame: Optional[np.ndarray] = None
        self.anchor_bounding_box: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.max_allowable_shift: float = 0.0
        self.logger = logging.getLogger(__name__)

    def set_anchor(self, points: np.ndarray):
        """
        Designate the highest-quality central frame as the immutable anchor.
        Calculates the absolute bounding box diagonal to establish the shift limit.
        
        Args:
            points: (N, 3) geometry array of the anchor pointcloud.
        """
        if points is None or len(points) == 0:
            raise ValueError("Anchor frame contains no valid points.")

        self.anchor_frame = points
        
        # Calculate bounding box
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        self.anchor_bounding_box = (min_bounds, max_bounds)
        
        # Absolute bounding box diagonal
        diagonal = np.linalg.norm(max_bounds - min_bounds)
        
        # Maximum allowable shift is 5% of the bounding box diagonal
        self.max_allowable_shift = diagonal * self.shift_limit_ratio
        self.logger.info(f"[Spatial Guard] Anchor set. Diagonal={diagonal:.4f}m | Max Shift={self.max_allowable_shift:.4f}m")

    def validate_transform(self, translation_vector: np.ndarray) -> bool:
        """
        Reject any transformation matrix that violates the strict translation limit.
        
        Args:
            translation_vector: The (x, y, z) shift attempted by the registration loop.
            
        Returns:
            True if the transform is within physical limits, False if it drifts catastrophically.
        """
        if self.anchor_frame is None:
            self.logger.error("[Spatial Guard] Anchor not set. Rejecting transform.")
            return False

        shift_magnitude = np.linalg.norm(translation_vector)
        
        if shift_magnitude > self.max_allowable_shift:
            self.logger.error(
                f"[Spatial Guard] REJECTED: Catastrophic drift detected! "
                f"Shift {shift_magnitude:.4f}m exceeds strict limit {self.max_allowable_shift:.4f}m."
            )
            # Conceptually triggers automatic hardware re-calibration here.
            return False
            
        return True

    def process_incoming_frame(self, state: Dict) -> bool:
        """
        Integration endpoint for SafetyKernel. Extracts translation hypothesis 
        from the state envelope and enforces constraints against the anchor.
        
        Args:
            state: The current robot state containing 'registration_shift'.
        Returns:
            True if secure, False if the frame poses a geometric hallucination risk.
        """
        shift = state.get("registration_shift")
        
        if shift is None:
            # Assumes no spatial update is occurring in this tick
            return True
            
        return self.validate_transform(np.array(shift))
