"""Step 4A.0.B — Verify PI-FNO checkpoint identity + metadata + instantiation.

Handles Phase 3 checkpoint key 'val_data_loss' and load_phase2_fno() signature.
"""
import json
import hashlib
import sys
import inspect
from pathlib import Path

sys.path.insert(0, "src")

import torch

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

EXPECTED_SHA = "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50"
CKPT_PATH = "models/sc_fno/phase_3/pi_fno/best_val.pt"

out = {"checks": {}, "all_pass": False, "metadata": {}}
def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 1. SHA-256
actual_sha = sha256_file(CKPT_PATH)
check("ckpt_sha256", actual_sha == EXPECTED_SHA, actual_sha[:16] + "...")

# 2. File size
size = Path(CKPT_PATH).stat().st_size
out["metadata"]["file_size_bytes"] = size
out["metadata"]["file_size_MB"] = round(size / 1e6, 3)
check("ckpt_exists", size > 0, f"{size} bytes")

# 3. Load and inspect
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
top_keys = list(ckpt.keys())
out["metadata"]["top_level_keys"] = top_keys
check("has_model_state_dict", "model_state_dict" in ckpt, str(top_keys))

# 4. Extract Phase 3's actual metadata keys
#    Phase 3 used: epoch, variant, padding, use_physics, val_data_loss, lam_phys
for k in ("epoch", "variant", "padding", "use_physics", "val_data_loss",
          "lam_phys", "val_loss", "best_val_loss", "config", "seed"):
    if k in ckpt:
        val = ckpt[k]
        if isinstance(val, (int, float, str, bool)):
            out["metadata"][k] = val
        elif isinstance(val, dict):
            out["metadata"][k] = {sk: str(sv) for sk, sv in val.items()}
        else:
            out["metadata"][k] = str(type(val))

# 5. Check for val_data_loss specifically (Phase 3's recorded key)
val_data_loss = ckpt.get("val_data_loss", None)
if val_data_loss is None:
    val_data_loss = ckpt.get("val_loss", ckpt.get("best_val_loss", None))
check("has_val_data_loss", val_data_loss is not None,
      f"val_data_loss = {val_data_loss}")
out["metadata"]["val_data_loss_resolved"] = val_data_loss

# 6. State dict analysis
sd = ckpt["model_state_dict"]
n_tensors = len([k for k in sd.keys() if k != "_metadata"])
out["metadata"]["state_dict_tensor_count"] = n_tensors
out["metadata"]["state_dict_has_metadata_key"] = "_metadata" in sd
check("state_dict_nonempty", n_tensors > 0, f"{n_tensors} tensors")

# 7. Confirm NOT P2-FNO checkpoint
p2_sha = sha256_file("models/baseline/fno_best.pt")
check("not_p2_ckpt", actual_sha != p2_sha,
      f"PI-FNO({actual_sha[:8]}) ≠ P2({p2_sha[:8]})")

# 8. Instantiate — inspect signature first
try:
    from physics_informed.derivatives import load_phase2_fno, SCFNOModel
    from physics_informed.decoder import build_decoder

    # Inspect load_phase2_fno signature
    sig = inspect.signature(load_phase2_fno)
    out["metadata"]["load_phase2_fno_signature"] = str(sig)
    print(f"  load_phase2_fno signature: {sig}")

    # Build kwargs from actual signature
    kwargs = {}
    params = sig.parameters
    if "device" in params:
        kwargs["device"] = "cpu"
    if "ckpt_path" in params:
        kwargs["ckpt_path"] = "models/baseline/fno_best.pt"
    elif "path" in params:
        kwargs["path"] = "models/baseline/fno_best.pt"
    # else: no path argument — function likely hardcodes the Phase 2 path

    fno = load_phase2_fno(**kwargs)
    decoder = build_decoder(seed=42).to("cpu")
    model = SCFNOModel(fno, decoder).to("cpu")

    sd_clean = {k: v for k, v in sd.items() if k != "_metadata"}
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    out["metadata"]["missing_keys"] = list(missing)
    out["metadata"]["unexpected_keys"] = list(unexpected)

    out["metadata"]["param_count_total"] = sum(p.numel() for p in model.parameters())
    check("param_count_63634", out["metadata"]["param_count_total"] == 63634,
          f"{out['metadata']['param_count_total']} params")

    # Forward pass on dummy batch
    model.eval()
    x = torch.randn(2, 1, 200)
    with torch.no_grad():
        y_data = model.fno(x)                      # [2, 9, 200]
        t_q = torch.linspace(0.5, 9.5, 10).unsqueeze(0).expand(2, -1).contiguous()
        # Try to find latent extractor name
        latent_fn = getattr(model, "_fno_latent", None)
        if latent_fn is None:
            latent_fn = getattr(model, "extract_latent", None)
        if latent_fn is None:
            raise AttributeError(
                "No _fno_latent or extract_latent method on SCFNOModel. "
                f"Available attrs: {[a for a in dir(model) if not a.startswith('__')]}"
            )
        z = latent_fn(x)                           # latent
        y_phys = model.decoder(z, t_q)             # [2, 9, 10]

    check("forward_data_shape", list(y_data.shape) == [2, 9, 200], str(y_data.shape))
    check("forward_phys_shape", list(y_phys.shape) == [2, 9, 10], str(y_phys.shape))
    check("forward_data_finite", bool(torch.isfinite(y_data).all()), "finite")
    check("forward_phys_finite", bool(torch.isfinite(y_phys).all()), "finite")

except Exception as e:
    check("instantiation", False, f"{type(e).__name__}: {e}")
    print(f"\n⚠ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()

out["all_pass"] = all(c["pass"] for c in out["checks"].values())

with open("results/phase_4/phase_4a/preflight/checkpoint_identity.json", "w") as f:
    json.dump(out, f, indent=2)
with open("results/phase_4/phase_4a/preflight/checkpoint_metadata.json", "w") as f:
    json.dump(out["metadata"], f, indent=2, default=str)

print(f"\n=== 4A.0.B Checkpoint identity: {'PASS' if out['all_pass'] else 'FAIL'} ===")
