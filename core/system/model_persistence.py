import pickle
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import numpy as np
import json
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl

class ModelPersister:
    """Enhanced model persistence with multiple formats"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    def save(self, agent: Any, metadata: Dict = None) -> Dict[str, Path]:
        """Save model in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"agent_{timestamp}"
        
        saved_paths = {
            'pkl': self._save_pickle(agent, base_name),
            'onnx': self._save_onnx(agent, base_name),
            'metadata': self._save_metadata(metadata, base_name)
        }
        
        self._create_latest_symlinks(base_name)
        return saved_paths

    def _save_pickle(self, agent, base_name: str) -> Path:
        """Save using stable-baselines3's safe serialization"""
        path = self.model_dir / f"{base_name}.pkl"
        with open(path, 'wb') as f:
            save_to_pkl(agent, f)
        return path

    def _save_onnx(self, agent, base_name: str) -> Path:
        """Export policy to ONNX format"""
        path = self.model_dir / f"{base_name}.onnx"
        
        # Get sample input shapes from agent
        market_shape = agent.observation_space['market'].shape
        portfolio_shape = agent.observation_space['portfolio'].shape
        
        # Create dummy inputs
        dummy_input = (
            {'market': torch.randn(1, *market_shape),
             'portfolio': torch.randn(1, *portfolio_shape)}
        )
        
        # Export the policy network
        torch.onnx.export(
            agent.policy,
            dummy_input,
            path,
            input_names=['market', 'portfolio'],
            output_names=['actions'],
            dynamic_axes={
                'market': {0: 'batch_size'},
                'portfolio': {0: 'batch_size'},
                'actions': {0: 'batch_size'}
            },
            opset_version=12,
            do_constant_folding=True
        )
        
        # Verify the exported model
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        
        return path

    def _save_metadata(self, metadata: Dict, base_name: str) -> Path:
        """Save training metadata as JSON"""
        if not metadata:
            return None
            
        path = self.model_dir / f"{base_name}_metadata.json"
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return path

    def _create_latest_symlinks(self, base_name: str):
        """Create symlinks to latest versions"""
        for ext in ['pkl', 'onnx']:
            latest = self.model_dir / f"latest.{ext}"
            latest.unlink(missing_ok=True)
            latest.symlink_to(f"{base_name}.{ext}")

    def load(self, model_type: str = 'pkl') -> Any:
        """Load model from preferred format"""
        if model_type == 'onnx':
            return self._load_onnx()
        return self._load_pickle()

    def _load_pickle(self) -> Any:
        """Load pickle-serialized model"""
        path = self.model_dir / "latest.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            return load_from_pkl(f)

    def _load_onnx(self) -> ort.InferenceSession:
        """Load ONNX model for inference"""
        path = self.model_dir / "latest.onnx"
        if not path.exists():
            raise FileNotFoundError(f"No ONNX model found at {path}")
            
        # Try GPU first, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ort.InferenceSession(str(path), providers=providers)