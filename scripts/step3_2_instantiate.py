"""Step 3.2.B — Instantiate SCFNO, verify Phase 2 SHA, measure decoder params."""
import hashlib, json, sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from physics_informed.decoder import build_decoder
from physics_informed.derivatives import load_phase2_fno, SCFNOModel


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

CKPT = "models/baseline/fno_best.pt"
EXPECTED_SHA = "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889"

sha_before = sha256(CKPT)
assert sha_before == EXPECTED_SHA, f"Phase 2 checkpoint SHA changed: {sha_before}"

# Instantiate
device = "cuda" if torch.cuda.is_available() else "cpu"
fno = load_phase2_fno(CKPT, device=device)
decoder = build_decoder(latent_dim=32, hidden_dim=64, out_dim=9,
                        n_freqs=16, T=10.0, seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device)

# Count params
fno_params = sum(p.numel() for p in model.fno.parameters() if p.requires_grad)
dec_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
total = fno_params + dec_params

# Save decoder init
dec_init_path = "models/sc_fno/phase_3/decoder_init.pt"
Path(dec_init_path).parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "decoder_state_dict": decoder.state_dict(),
    "decoder_config": {
        "latent_dim": 32, "hidden_dim": 64, "out_dim": 9,
        "n_freqs": 16, "T": 10.0, "seed": 42,
    },
}, dec_init_path)
dec_init_size = Path(dec_init_path).stat().st_size

# SHA after — must be identical (Phase 2 checkpoint untouched)
sha_after = sha256(CKPT)

spec = {
    "frozen_spec": {
        "type": "temporal-attention coordinate decoder",
        "latent_dim": 32, "hidden_dim": 64, "out_dim": 9,
        "n_freqs": 16, "T_seconds": 10.0,
        "query_embed_dim": 2*16+1,
        "n_heads": 4,
        "init_seed": 42,
    },
    "param_counts": {
        "fno_backbone": fno_params,
        "decoder": dec_params,
        "total": total,
    },
    "checkpoint": {
        "phase2_sha_before": sha_before,
        "phase2_sha_after": sha_after,
        "phase2_unchanged": sha_before == sha_after,
        "decoder_init_path": dec_init_path,
        "decoder_init_size_bytes": dec_init_size,
    },
    "device": device,
}
out_path = "results/phase_3/derivatives/decoder_specification.json"
Path(out_path).write_text(json.dumps(spec, indent=2))
print(json.dumps(spec, indent=2))
print("\nPhase 2 SHA unchanged:", sha_before == sha_after)
print(f"FNO params: {fno_params:,}")
print(f"Decoder params: {dec_params:,}")
print(f"Total: {total:,}")
assert sha_before == sha_after, "CRITICAL: Phase 2 checkpoint modified"
