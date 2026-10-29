import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eld_ai.solver import solve_eld

def test_lossless_balance_and_limits():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config.json").read_text())
    sol = solve_eld(800.0, cfg["generators"], cfg["losses"])
    assert abs(sol["balance_error_mw"]) < 1e-3
    for p, g in zip(sol["dispatch_mw"], cfg["generators"]):
        assert g["pmin"] - 1e-6 <= p <= g["pmax"] + 1e-6
