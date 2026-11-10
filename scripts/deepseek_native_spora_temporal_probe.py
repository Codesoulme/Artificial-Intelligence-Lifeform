"""Compare the native SPORA DeltaMemorySite reader on the temporal pilot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_temporal_intent_router import run


if __name__ == '__main__':
    run(native_spora=True)
