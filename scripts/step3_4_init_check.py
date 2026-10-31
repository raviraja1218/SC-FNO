"""Step 3.4.B — Verify Phase 2 + decoder init SHAs before training."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
from physics_informed.decoder import build_decoder
from physics_informed.derivatives import load_phase2_fno

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

EXPECTED_FNO = "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889"
EXPECTED_DEC = "c30dd6ca0cc7fb87918696265285eb041f8c214264c1ab6622659fd6623c4b3f"

assert sha("models/baseline/fno_best.pt") == EXPECTED_FNO
assert sha("models/sc_fno/phase_3/decoder_init.pt") == EXPECTED_DEC

print(f"Phase 2 ckpt SHA: {EXPECTED_FNO[:16]}...  OK")
print(f"Decoder init SHA: {EXPECTED_DEC[:16]}...  OK")
print("=== 3.4.B PASS ===")
