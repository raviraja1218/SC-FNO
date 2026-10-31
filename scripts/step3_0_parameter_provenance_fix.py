"""Fix Step 3.0.C — extract physical parameters, robustly handling YAML types."""
import csv
import json
import yaml

kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg  = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))

def scalar(v):
    """Coerce YAML value to float. Handles list, dict, str, int, float."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return float(v[0])
        raise ValueError(f"Expected scalar, got list of {len(v)}: {v}")
    if isinstance(v, dict):
        raise ValueError(f"Expected scalar, got dict: {v}")
    return float(v)

rows = []

def add(group, name, value, unit, source, note=""):
    rows.append({"group": group, "parameter": name, "value": value,
                 "unit": unit, "source": source, "note": note})

# --- Kinetics ---
n = kin_cfg["neutronics"]
add("kinetics", "n_precursor_groups", scalar(n["n_precursor_groups"]), "—",
    "reactor_parameters.v3.yaml:neutronics.n_precursor_groups")
add("kinetics", "beta_eff", scalar(n["beta_eff"]), "dimensionless",
    "reactor_parameters.v3.yaml:neutronics.beta_eff")
add("kinetics", "Lambda", scalar(n["Lambda"]), "s",
    "reactor_parameters.v3.yaml:neutronics.Lambda")

beta_sum = 0.0
for g in n["precursor_groups"]:
    i = scalar(g["i"]) if "i" in g else None
    b = scalar(g["beta_i"])
    l = scalar(g["lambda_i"])
    add("kinetics", f"beta_{i}", b, "dimensionless",
        f"reactor_parameters.v3.yaml:neutronics.precursor_groups[i={i}].beta_i")
    add("kinetics", f"lambda_{i}", l, "s^-1",
        f"reactor_parameters.v3.yaml:neutronics.precursor_groups[i={i}].lambda_i")
    beta_sum += b

beta_eff = scalar(n["beta_eff"])
add("kinetics", "sum_beta_check", round(beta_sum, 10), "dimensionless",
    "computed", f"Σβᵢ = {beta_sum:.10f}, |Σβᵢ − β_eff| = {abs(beta_sum - beta_eff):.3e}")

ext = n["external_source"]
add("kinetics", "include_q_t", bool(ext["include_q_t"]), "bool",
    "reactor_parameters.v3.yaml:neutronics.external_source.include_q_t",
    "Phase 0 decision: q(t) INCLUDED")
add("kinetics", "q_t_default", scalar(ext["q_t_default"]), "—",
    "reactor_parameters.v3.yaml:neutronics.external_source.q_t_default")

init = kin_cfg["initial_state"]
add("kinetics", "n0", scalar(init["n0"]), "—", "reactor_parameters.v3.yaml:initial_state.n0")
add("kinetics", "T_f0", scalar(init["T_f0"]), "K", "reactor_parameters.v3.yaml:initial_state.T_f0")
add("kinetics", "T_c0", scalar(init["T_c0"]), "K", "reactor_parameters.v3.yaml:initial_state.T_c0")
add("kinetics", "C_i0_formula", "beta_i*n0/(Lambda*lambda_i)", "—",
    "reactor_parameters.v3.yaml:initial_state (implicit)",
    "equilibrium precursor concentration")

# --- Thermal-hydraulics ---
fuel = th_cfg["fuel"]; coolant = th_cfg["coolant"]; ht = th_cfg["heat_transfer"]
power = th_cfg["power"]; fb = th_cfg["reactivity_feedback"]

for k, unit in [("M_f","kg"),("c_p_f","J/(kg*K)"),("f_f","dimensionless")]:
    add("fuel", k, scalar(fuel[k]), unit, f"thermal_hydraulics.v4.yaml:fuel.{k}")

for k, unit in [("M_c","kg"),("c_p_c","J/(kg*K)"),("m_dot","kg/s"),("T_in","K")]:
    add("coolant", k, scalar(coolant[k]), unit, f"thermal_hydraulics.v4.yaml:coolant.{k}")

add("heat_transfer", "U", scalar(ht["U"]), "W/(m^2*K)", "thermal_hydraulics.v4.yaml:heat_transfer.U")
add("heat_transfer", "A", scalar(ht["A"]), "m^2", "thermal_hydraulics.v4.yaml:heat_transfer.A")
add("heat_transfer", "UA", scalar(ht["U"]) * scalar(ht["A"]), "W/K", "computed",
    f"U*A = {scalar(ht['U'])*scalar(ht['A']):.3e}")

add("power", "P0", scalar(power["P0"]), "W", "thermal_hydraulics.v4.yaml:power.P0")

for k, unit in [("alpha_f","(Δk/k)/K"),("alpha_c","(Δk/k)/K"),
                ("T_f_ref","K"),("T_c_ref","K")]:
    add("feedback", k, scalar(fb[k]), unit, f"thermal_hydraulics.v4.yaml:reactivity_feedback.{k}")

# --- Equilibrium check ---
UA    = scalar(ht["U"]) * scalar(ht["A"])
P0    = scalar(power["P0"])
f_f   = scalar(fuel["f_f"])
T_f0  = scalar(init["T_f0"])
T_c0  = scalar(init["T_c0"])
m_dot = scalar(coolant["m_dot"])
c_p_c = scalar(coolant["c_p_c"])
T_in  = scalar(coolant["T_in"])
n0    = scalar(init["n0"])

fuel_heating      = f_f * P0 * n0
heat_transfer_out = UA * (T_f0 - T_c0)
coolant_balance   = P0 - 2.0 * m_dot * c_p_c * (T_c0 - T_in)

add("equilibrium", "fuel_heating_W", fuel_heating, "W", "computed",
    f"f_f*P0*n0 = {fuel_heating:.3e}")
add("equilibrium", "heat_transfer_out_W", heat_transfer_out, "W", "computed",
    f"UA*(T_f0-T_c0) = {heat_transfer_out:.3e}")
add("equilibrium", "fuel_imbalance_ratio",
    round(fuel_heating / heat_transfer_out, 6) if heat_transfer_out != 0 else None,
    "dimensionless", "computed", "should be ≈ 1.0")
add("equilibrium", "coolant_balance_W", coolant_balance, "W", "computed",
    f"P0 - 2*mdot*cp_c*(Tc0-Tin) = {coolant_balance:.3e}")

with open("results/phase_3/preflight/parameter_provenance.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group","parameter","value","unit","source","note"])
    w.writeheader()
    w.writerows(rows)

n_params = sum(1 for r in rows if r["group"] != "equilibrium")
ratio = fuel_heating / heat_transfer_out if heat_transfer_out else 0
summary = {
    "total_params": n_params,
    "groups": {},
    "fuel_equilibrium_ratio": round(ratio, 6),
    "fuel_equilibrium_verified": abs(ratio - 1.0) < 0.01,
    "coolant_balance_W": coolant_balance,
    "all_pass": abs(ratio - 1.0) < 0.01,
}
for r in rows:
    summary["groups"].setdefault(r["group"], 0)
    summary["groups"][r["group"]] += 1

with open("results/phase_3/preflight/parameter_provenance.json", "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=2, default=str)

print(json.dumps(summary, indent=2))
print(f"\nParameter rows: {len(rows)}  (non-equilibrium params: {n_params})")
print("=== Parameter provenance:", "PASS" if summary["all_pass"] else "CHECK", "===")
