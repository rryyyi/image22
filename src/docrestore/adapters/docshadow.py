from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


class DocShadowRunner:
    """Optional adapter; the official model remains in third_party, not in our package."""
    def __init__(self, repository: str | Path, weights: str | Path, device: str | None = None):
        self.repository = Path(repository).resolve()
        self.weights = Path(weights).resolve()
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))
        import torch
        from models import Model
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Model()
        checkpoint = torch.load(self.weights, map_location=self.device, weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device) / 255.0
        with torch.inference_mode():
            prediction = self.model(tensor).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(np.rint(prediction * 255), 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
