"""Step 3.0.H — Confirm q(t) source term handling."""
import json
import yaml

kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
ext = kin_cfg["neutronics"]["external_source"]

out = {
    "include_q_t": bool(ext["include_q_t"]),
    "q_t_default": float(ext["q_t_default"]),
    "phase0_decision": "q(t) INCLUDED per Phase 0 lock e7e70c0",
    "provenance": "configs/reactor_parameters.v3.yaml → neutronics.external_source",
    "note": "Residual implementation in Step 3.1 must use q(t) per this config. "
            "Do NOT silently omit or introduce a new source model.",
    "all_pass": bool(ext["include_q_t"]) and abs(float(ext["q_t_default"])) < 1e-12,
}
with open("results/phase_3/preflight/source_term_confirmation.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("\n=== Source term:", "PASS" if out["all_pass"] else "CHECK", "===")
