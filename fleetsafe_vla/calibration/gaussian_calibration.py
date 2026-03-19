"""
gaussian_calibration.py — SOTA 3DGS Auto-Calibrator
Adapted from standard Multi-View 3D Rendering Python scripts.

Orchestrates automated hemispherical camera viewpoint generation and
formats physical/simulated imagery into identical transforms.json structures
for instantaneous 3D Gaussian Splatting / NeRF reconstruction.
"""
import math
import json
import os
import numpy as np
from typing import List, Dict

class AutoCalibrator:
    def __init__(self, output_dir: str = "logs/3dgs_datasets/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def compute_hemispherical_poses(self, anchor: np.ndarray, radius: float, num_cameras: int = 24) -> List[np.ndarray]:
        """
        Mimics automated Blender scripts: Distributes cameras spherically around an anchor.
        Computes 4x4 homogenous transformation matrices representing camera extrinsics.
        """
        poses = []
        # Uses Golden Spiral distribution over a hemisphere for uniform view coverage
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(num_cameras):
            # Calculate spherical coordinates
            theta = 2 * math.pi * i / golden_ratio # Azimuth
            phi = math.acos(1 - (i + 0.5) / num_cameras) # Zenith (0 to pi/2 for hemisphere)
            
            # Cartesian coordinates relative to anchor
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            
            camera_pos = anchor + np.array([x, y, z])
            
            # Compute look-at matrix (Z-axis points at anchor, Y-axis is arbitrary 'up')
            forward = anchor - camera_pos
            forward = forward / (np.linalg.norm(forward) + 1e-6)
            
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            
            # Resolve gimbal lock if looking straight down/up
            if np.linalg.norm(right) < 1e-4:
                up = np.array([0, 1, 0])
                right = np.cross(forward, up)
                
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # 3DGS / COLMAP usually expects OpenCV/OpenGL convention coordinates
            # Standard OpenGL: forward is -Z
            rot_matrix = np.array([
                [right[0], up[0], -forward[0]],
                [right[1], up[1], -forward[1]],
                [right[2], up[2], -forward[2]]
            ])
            
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transform[:3, 3] = camera_pos
            
            poses.append(transform)
            
        return poses
    
    def generate_transforms_json(self, session_id: str, camera_angle_x: float, camera_angle_y: float, image_files: List[str], poses: List[np.ndarray]):
        """
        Formats and dumps the classical transforms.json required to train 3DGS automatically.
        """
        if len(image_files) != len(poses):
            raise ValueError("Mismatched image files and poses.")
            
        dataset = {
            "camera_angle_x": camera_angle_x,
            "camera_angle_y": camera_angle_y,
            "fl_x": 500.0, # Approximate focal lengths
            "fl_y": 500.0,
            "cx": 400.0,
            "cy": 400.0,
            "w": 800,
            "h": 800,
            "frames": []
        }
        
        for idx, (img_path, pose) in enumerate(zip(image_files, poses)):
            frame = {
                "file_path": img_path,
                "rotation": 0.0,
                "transform_matrix": pose.tolist()
            }
            dataset["frames"].append(frame)
            
        out_path = os.path.join(self.output_dir, f"transforms_{session_id}.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=4)
            
        print(f"[AutoCalibrator] ✅ Automated 3DGS dataset generated at {out_path} with {len(poses)} multi-view frames.")
        return out_path
