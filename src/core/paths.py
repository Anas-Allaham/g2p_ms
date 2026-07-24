"""Single project-root anchor so relocated modules resolve
model/, data/, and g2p_pipeline_split_v2/ correctly."""

from pathlib import Path

# src/core/paths.py -> parents[2] == project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
