"""Quality-aware document restoration for local CPU/GPU environments."""

from .config import PipelineConfig, load_config
from .pipeline import DocumentPipeline

__all__ = ["DocumentPipeline", "PipelineConfig", "load_config"]
__version__ = "0.1.0"
