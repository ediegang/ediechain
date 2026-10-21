from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]

CONFIGS_DIR = REPO_DIR / "configs"

DATA_DIR = REPO_DIR / ".artifacts"
PIPELINE_DIR = DATA_DIR / "pipeline"
CACHE_DIR = DATA_DIR / "cache"
