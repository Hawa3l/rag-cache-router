from pathlib import Path

LARGE_MODEL = "nvidia/nemotron-3.5-lightning:free"
SMALL_MODEL = "liquid/lfm-2.5-2.6b:free"
PRICES = {
    LARGE_MODEL : (0.06, 0.35),
    SMALL_MODEL : (0.0005, 0.20)
}
K = 3
MAX_TOKENS = 50000
SIMILARITY_THRESHOLD = 0.3
CE_CUTOFF = 0.6
BASE_DIR = Path(__file__).parent
