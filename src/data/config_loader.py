"""Config resolution: load the highest-versioned YAML for a given base name.

Phase 0 locked reactor_parameters.v2.yaml. Phase 1 correction introduced
reactor_parameters.v3.yaml. This loader ensures all downstream modules use the
newest available version while preserving the locked files untouched.
"""
from pathlib import Path
import re
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / 'configs'


def resolve_config(base_name: str) -> Path:
    """Return the highest-versioned file for base_name.

    base_name examples: 'reactor_parameters', 'thermal_hydraulics'
    Matches: {base}.yaml and {base}.vN.yaml
    Returns: path to highest N (or base file if no versioned variant exists)
    """
    candidates = []
    base_file = CONFIG_DIR / f"{base_name}.yaml"
    if base_file.exists():
        candidates.append((0, base_file))

    pattern = re.compile(rf"^{re.escape(base_name)}\.v(\d+)\.yaml$")
    for p in CONFIG_DIR.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(f"No config found for base '{base_name}' in {CONFIG_DIR}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_config(base_name: str) -> dict:
    path = resolve_config(base_name)
    with open(path) as f:
        return yaml.safe_load(f)


def config_provenance(base_name: str) -> str:
    return str(resolve_config(base_name))
