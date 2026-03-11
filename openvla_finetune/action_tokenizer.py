"""
OpenVLA Action Tokenizer Rewrite for FLEET-Safe
-----------------------------------------------
Original OpenVLA outputs 7 discrete tokens representing Delta XYZ, Roll, Pitch, Yaw, and Gripper.
For a State-Of-The-Art (SOTA) Hospital Navigation Benchmark, we must completely overhaul 
this quantization logic.

This module intercepts the Llama-2 decoder outputs from OpenVLA and translates the discrete
bins into:
1. FastBot (Differential Drive): [v, omega]
2. Unitree G1 (Humanoid): [23 Joint Angles]
"""

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

class FleetActionTokenizer:
    def __init__(self, embodiment: str, num_bins: int = 256):
        """
        Initializes the action tokenizer.
        :param embodiment: 'fastbot' or 'g1'
        :param num_bins: The discrete vocabulary size allocated for action tokens in OpenVLA (usually 256).
        """
        self.embodiment = embodiment
        self.num_bins = num_bins
        
        # Define physical limits for de-quantization mapping
        if self.embodiment == 'fastbot':
            self.action_dim = 2
            # [linear velocity v (m/s), angular velocity w (rad/s)]
            self.action_min = np.array([-0.5, -1.0])
            self.action_max = np.array([1.5, 1.0])
        elif self.embodiment == 'g1':
            self.action_dim = 23
            # Unitree G1 23-DoF joint limits (simplified array mapping -pi to pi for joints)
            self.action_min = np.full(self.action_dim, -np.pi)
            self.action_max = np.full(self.action_dim, np.pi)
        else:
            raise ValueError(f"Unknown embodiment: {embodiment}. Must be 'fastbot' or 'g1'.")

    def tokenize_action(self, continuous_actions: np.ndarray) -> np.ndarray:
        """
        Used during Dataset Preparation: 
        Converts continuous ROS2/Sim ground-truth actions into discrete OpenVLA vocabulary bins.
        """
        continuous_actions = np.clip(continuous_actions, self.action_min, self.action_max)
        
        # Normalize to [0, 1]
        normalized_actions = (continuous_actions - self.action_min) / (self.action_max - self.action_min)
        
        # Quantize to bins (0 to num_bins - 1)
        discrete_actions = (normalized_actions * (self.num_bins - 1)).astype(int)
        
        return discrete_actions

    def detokenize_action(self, discrete_tokens: torch.Tensor) -> torch.Tensor:
        """
        Used during Evaluation and Deployment:
        Converts the Llama-2 discrete token outputs back into smooth continuous physical actions.
        """
        device = discrete_tokens.device
        
        # Ensure tokens are within bounds
        discrete_tokens = torch.clamp(discrete_tokens, 0, self.num_bins - 1).float()
        
        # Normalize to [0, 1]
        normalized_actions = discrete_tokens / (self.num_bins - 1)
        
        # Project back to physical bounds
        action_min_tensor = torch.tensor(self.action_min, device=device, dtype=torch.float32)
        action_max_tensor = torch.tensor(self.action_max, device=device, dtype=torch.float32)
        
        continuous_actions = normalized_actions * (action_max_tensor - action_min_tensor) + action_min_tensor
        return continuous_actions

    def patch_openvla_processor(self, processor):
        """
        Patches the Hugging Face OpenVLA Processor to utilize this custom tokenizer
        instead of the default Prismatic 7-DoF logic.
        """
        # Overwrite the internal detokenizer logic inside the pre-trained processor
        processor.action_tokenizer = self
        processor.tokenizer.add_tokens([f"<action_{i}>" for i in range(self.num_bins)])
        print(f"[*] Patched OpenVLA Processor for SOTA Embodiment: {self.embodiment}")
        return processor

# Example Usage
if __name__ == "__main__":
    print("Testing SOTA OpenVLA FastBot Tokenizer Re-write...")
    fastbot_tokenizer = FleetActionTokenizer('fastbot')
    
    # Simulate a ground truth FastBot ROS2 cmd_vel message (v=1.0 m/s, w=-0.2 rad/s)
    ground_truth = np.array([1.0, -0.2])
    print(f"Ground Truth FastBot action: {ground_truth}")
    
    # Tokenize for LLM injection
    tokens = fastbot_tokenizer.tokenize_action(ground_truth)
    print(f"Quantized OpenVLA Llama-2 Tokens: {tokens}")
    
    # Detokenize back
    recovered = fastbot_tokenizer.detokenize_action(torch.tensor(tokens))
    print(f"Recovered Continuous Action: {recovered.numpy()[:2]}")
