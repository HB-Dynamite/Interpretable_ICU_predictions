# this script is mainly concerned with path management as of now
from pathlib import Path
import pandas as pd

# root path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# definition of commo
DATA_DIR = PROJECT_ROOT / "data"
UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"
RESULTS_DIR = PROJECT_ROOT / "results"
HPO_DIR = PROJECT_ROOT / "hpo_configs"
LOG_DIR = PROJECT_ROOT / "logging"
