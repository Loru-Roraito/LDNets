from datetime import datetime
from pathlib import Path

START = datetime(1997, 4, 1) # date with first tide data

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"