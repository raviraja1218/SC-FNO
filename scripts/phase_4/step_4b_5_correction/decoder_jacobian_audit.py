"""Audit whether CoordinateDecoder is query-independent (per-position) or query-coupled (attention).

If query-independent: current per-channel derivative with grad_outputs=ones is CORRECT.
If query-coupled: current implementation is mathematically wrong; requires Jacobian diagonal.
"""
import sys, inspect
sys.path.insert(0, "src")

import torch

from physics_informed.decoder import CoordinateDecoder, build_decoder

print("=" * 70)
print("Decoder Jacobian Audit")
print("=" * 70)

# 1. Instantiate decoder
decoder = build_decoder(seed=42)
print(f"\nDecoder class: {type(decoder).__name__}")
print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")

# 2. Inspect forward signature
print(f"\nForward signature: {inspect.signature(decoder.forward)}")

# 3. Inspect source for attention / coupling
import inspect
src = inspect.getsource(decoder.forward)
print(f"\nForward source (first 50 lines):")
for i, line in enumerate(src.split("\n")[:50], 1):
    print(f"  {i:3d}: {line}")

# 4. Print all submodules
print(f"\nSubmodules:")
for name, mod in decoder.named_children():
    print(f"  {name}: {type(mod).__name__}")
    for sub_name, sub_mod in mod.named_children():
        print(f"    {sub_name}: {type(sub_mod).__name__}")

# 5. Empirical test: does query k output depend on query i≠k inputs?
print("\n" + "=" * 70)
print("Empirical coupling test")
print("=" * 70)
decoder.eval()

B, Nq, z_dim = 2, 10, 32
z = torch.randn(B, z_dim, 200)   # latent
t_q = torch.linspace(0.5, 9.5, Nq).unsqueeze(0).expand(B, -1).clone()

with torch.no_grad():
    out_a = decoder(z, t_q)   # [B, 9, Nq]

    # Perturb t_q at query 5 only
    t_q_perturbed = t_q.clone()
    t_q_perturbed[:, 5] += 0.01
    out_b = decoder(z, t_q_perturbed)

    diff = (out_b - out_a).abs()  # [B, 9, Nq]

    # Check: does output at queries != 5 change?
    print(f"\nPerturbed t_q[5] by +0.01")
    print(f"Output change at query 5:       {diff[:, :, 5].mean().item():.6e}")
    print(f"Output change at queries i≠5:")
    mask = torch.ones(Nq, dtype=torch.bool)
    mask[5] = False
    other_change = diff[:, :, mask].mean().item()
    print(f"  Mean |Δu| at i≠5:              {other_change:.6e}")
    print(f"  Max |Δu| at i≠5:               {diff[:, :, mask].max().item():.6e}")

    if other_change < 1e-6:
        print("\n>>> VERDICT: Query-INDEPENDENT")
        print(">>> Current per-channel derivative with grad_outputs=ones is CORRECT")
        print(">>> ∂u_j(t_k)/∂t_k = ∂u_j(t_k)/∂t |_{t_i fixed}")
    else:
        print(f"\n>>> VERDICT: Query-COUPLED (attention)")
        print(f">>> Current implementation OVER-COUNTS by sum over query positions")
        print(f">>> Requires torch.func.jacrev to extract Jacobian diagonal")
        print(f">>> Coupling magnitude: {other_change:.2e}")

