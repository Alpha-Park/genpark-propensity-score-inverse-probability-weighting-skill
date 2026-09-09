"""
MCP Server for Propensity Score Inverse Probability Weighting (IPW) Skill
"""

import json
import sys
from client import PropensityScoreIPW

def handle_call(name: str, args: dict) -> dict:
    if name == "estimate_ate_ipw":
        x = args.get("covariate_x", [0.0, 1.0, -1.0])
        t = args.get("treatment_t", [0, 1, 0])
        y = args.get("outcome_y", [1.0, 6.0, 0.5])
        ipw = PropensityScoreIPW()
        res = ipw.estimate_ate(x, t, y)
        return res
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
