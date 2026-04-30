import json
from pathlib import Path

outdir = Path("/data/manikm/manik/normbench_v06/results")
files = sorted(outdir.glob("normbench_v06_summary_*.json"))
if not files:
    raise SystemExit("No summary files found.")
latest = files[-1]
data = json.loads(latest.read_text(encoding="utf-8"))
print(f"Latest summary: {latest}\n")
for row in data[:40]:
    print(
        f"{row['model']:12s} | {row['environment']:20s} | {row['role_mode']:14s} | "
        f"{row['prompt_variant']:2s} | {row['condition']:18s} | runs={row['runs']:2d} | "
        f"mean_uB-uA={row['mean_uB_minus_uA']:6.2f} | "
        f"mean_valid_rep={row['mean_valid_reports']:5.2f} | "
        f"mean_over={row['mean_over_take_violations']:5.2f} | "
        f"std_uB-uA={row['std_uB_minus_uA']:5.2f}"
    )
