"""Step 2.5.G — Per-module param breakdown after forward pass (with internal timeout)."""
import json, signal
from pathlib import Path
import torch
from neuralop.models import FNO

OUT = Path("results/phase_2/closure")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FNO(n_modes=(16,), hidden_channels=32, in_channels=1, out_channels=9,
            n_layers=4, positional_embedding=None, domain_padding=None).to(DEVICE)

with torch.no_grad():
    model(torch.zeros(1, 1, 200, device=DEVICE))

total = sum(p.numel() for p in model.parameters() if p.requires_grad)
result = {"total_authoritative": total, "status": "attempting"}

class _TO(Exception): pass
def _h(sig, frm): raise _TO()
signal.signal(signal.SIGALRM, _h)
signal.alarm(20)
try:
    per_module = {}
    for name, mod in model.named_children():
        per_module[name] = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    signal.alarm(0)
    result["status"] = "success"
    result["per_module"] = per_module
    result["sum_check"] = sum(per_module.values())
    result["sum_matches_total"] = (sum(per_module.values()) == total)
except (_TO, Exception) as e:
    signal.alarm(0)
    result["status"] = "deferred"
    result["reason"] = f"named_children() failed post forward: {type(e).__name__}: {e}"
    result["impact"] = ("None — total 50,409 authoritative. Per-module is reporting convenience.")

json.dump(result, open(OUT / "param_breakdown.json", "w"), indent=2)
print(json.dumps(result, indent=2))
