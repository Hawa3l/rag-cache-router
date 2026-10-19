LARGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
SMALL_MODEL = "liquid/lfm-2.5-2.6b:free"
PRICES = {
    LARGE_MODEL : (0.085, 0.40),
    SMALL_MODEL : (0.0005, 0.20)
}
K = 3
MAX_TOKENS = 50000
SIMILARITY_THRESHOLD = 0.9
