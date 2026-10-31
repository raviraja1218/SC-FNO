"""Step 3.2.I — Measure GPU memory and runtime for compute_du_dt."""
import json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "src")
from physics_informed.decoder import build_decoder
from physics_informed.derivatives import (
    load_phase2_fno, SCFNOModel, compute_du_dt,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
fno = load_phase2_fno(device=device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device).train()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

B, Nq = 16, 200
rho = torch.randn(B, 1, 200, device=device) * 1e-3
t_q = torch.linspace(0.5, 9.5, Nq, device=device).repeat(B, 1)

# Warm-up
for _ in range(3):
    z, u, du = compute_du_dt(model, rho, t_q)
    (du**2).mean().backward()
    model.zero_grad()
torch.cuda.synchronize()

# Timed
N_rep = 10
t0 = time.perf_counter()
for _ in range(N_rep):
    z, u, du = compute_du_dt(model, rho, t_q)
    L = (du**2).mean()
    L.backward()
    model.zero_grad()
torch.cuda.synchronize()
t_elapsed = time.perf_counter() - t0

peak_alloc = torch.cuda.max_memory_allocated() / 1e6
peak_reserved = torch.cuda.max_memory_reserved() / 1e6

out = {
    "device": device,
    "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
    "batch_size": B,
    "n_query": Nq,
    "n_repeats": N_rep,
    "total_wall_time_s": round(t_elapsed, 4),
    "per_iter_ms": round(t_elapsed / N_rep * 1000, 2),
    "peak_allocated_MB": round(peak_alloc, 2),
    "peak_reserved_MB": round(peak_reserved, 2),
    "three_GB_target_MB": 3000,
    "under_3GB": peak_alloc < 3000,
}
Path("results/phase_3/derivatives/gpu_runtime.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
