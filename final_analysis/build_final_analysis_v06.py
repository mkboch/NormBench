import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev

RUNS = Path("/data/manikm/manik/normbench_v06/results/normbench_v06_runs_1777118963.json")
OUT = Path("/data/manikm/manik/normbench_v06/final_analysis")
OUT.mkdir(parents=True, exist_ok=True)

rows = json.loads(RUNS.read_text())

def filt(model=None, env=None, role=None, prompt=None, cond=None):
    out = []
    for r in rows:
        if model is not None and r["model"] != model:
            continue
        if env is not None and r["environment"] != env:
            continue
        if role is not None and r["role_mode"] != role:
            continue
        if prompt is not None and r["prompt_variant"] != prompt:
            continue
        if cond is not None and r["condition"] != cond:
            continue
        out.append(r)
    return out

def vals(sub, metric):
    return [x[metric] for x in sub]

def summarize(sub, metric):
    v = vals(sub, metric)
    return mean(v), (pstdev(v) if len(v) > 1 else 0.0)

def cohend(a, b):
    ma, mb = mean(a), mean(b)
    va = sum((x-ma)**2 for x in a) / max(1, len(a)-1)
    vb = sum((x-mb)**2 for x in b) / max(1, len(b)-1)
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(1, (len(a)+len(b)-2)))
    if pooled == 0:
        return float("inf") if ma != mb else 0.0
    return (ma - mb) / pooled

def perm_p(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    observed = abs(mean(a) - mean(b))
    both = a + b
    count = 0
    for _ in range(n):
        rng.shuffle(both)
        aa = both[:len(a)]
        bb = both[len(a):]
        if abs(mean(aa) - mean(bb)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n + 1)

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def write_md_table(path, rows):
    if not rows:
        path.write_text("No rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

metrics = ["uB_minus_uA", "over_take_violations", "valid_reports", "report_precision", "violation_rate_per_round"]

# 1. Main table: Qwen resource v3 across role modes and conditions
main_rows = []
for role in ["explicit_roles", "light_roles", "uniform", "random"]:
    prompt = "v3" if role != "random" else "v1"
    for cond in ["NORMAL", "TEMPTATION", "ENFORCEMENT", "COSTLY_ENFORCEMENT", "NOISY_ENFORCEMENT"]:
        sub = filt(model="qwen3:8b", env="resource_two_stage", role=role, prompt=prompt, cond=cond)
        if not sub:
            continue
        row = {"model":"qwen3:8b","environment":"resource_two_stage","role_mode":role,"prompt_variant":prompt,"condition":cond,"runs":len(sub)}
        for m in metrics:
            mu, sd = summarize(sub, m)
            row[f"mean_{m}"] = round(mu, 4)
            row[f"sd_{m}"] = round(sd, 4)
        main_rows.append(row)

# 2. Prompt sensitivity: Qwen explicit roles resource
prompt_rows = []
for pv in ["v1","v2","v3"]:
    for cond in ["NORMAL", "TEMPTATION", "ENFORCEMENT", "COSTLY_ENFORCEMENT", "NOISY_ENFORCEMENT"]:
        sub = filt(model="qwen3:8b", env="resource_two_stage", role="explicit_roles", prompt=pv, cond=cond)
        row = {"model":"qwen3:8b","environment":"resource_two_stage","role_mode":"explicit_roles","prompt_variant":pv,"condition":cond,"runs":len(sub)}
        for m in metrics:
            mu, sd = summarize(sub, m)
            row[f"mean_{m}"] = round(mu, 4)
            row[f"sd_{m}"] = round(sd, 4)
        prompt_rows.append(row)

# 3. Second norm family
turn_rows = []
for role in ["explicit_roles", "light_roles", "uniform", "random"]:
    prompt = "v3" if role != "random" else "v1"
    for cond in ["NORMAL", "TEMPTATION", "ENFORCEMENT", "COSTLY_ENFORCEMENT", "NOISY_ENFORCEMENT"]:
        sub = filt(model="qwen3:8b", env="turn_taking_two_stage", role=role, prompt=prompt, cond=cond)
        if not sub:
            continue
        row = {"model":"qwen3:8b","environment":"turn_taking_two_stage","role_mode":role,"prompt_variant":prompt,"condition":cond,"runs":len(sub)}
        for m in ["uB_minus_uA", "valid_reports", "report_precision", "violation_rate_per_round"]:
            mu, sd = summarize(sub, m)
            row[f"mean_{m}"] = round(mu, 4)
            row[f"sd_{m}"] = round(sd, 4)
        turn_rows.append(row)

# 4. Llama robustness
llama_rows = []
for role in ["explicit_roles", "light_roles", "uniform", "random"]:
    prompt = "v3" if role != "random" else "v1"
    for cond in ["NORMAL", "TEMPTATION", "ENFORCEMENT", "COSTLY_ENFORCEMENT", "NOISY_ENFORCEMENT"]:
        sub = filt(model="llama3.1:8b", env="resource_two_stage", role=role, prompt=prompt, cond=cond)
        if not sub:
            continue
        row = {"model":"llama3.1:8b","environment":"resource_two_stage","role_mode":role,"prompt_variant":prompt,"condition":cond,"runs":len(sub)}
        for m in metrics:
            mu, sd = summarize(sub, m)
            row[f"mean_{m}"] = round(mu, 4)
            row[f"sd_{m}"] = round(sd, 4)
        llama_rows.append(row)

# 5. Pairwise tests
tests = [
    ("Q1","explicit ENF vs light ENF","uB_minus_uA",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","ENFORCEMENT"),
     filt("qwen3:8b","resource_two_stage","light_roles","v3","ENFORCEMENT")),
    ("Q2","explicit ENF vs uniform ENF","uB_minus_uA",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","ENFORCEMENT"),
     filt("qwen3:8b","resource_two_stage","uniform","v3","ENFORCEMENT")),
    ("Q3","explicit ENF vs random ENF","uB_minus_uA",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","ENFORCEMENT"),
     filt("qwen3:8b","resource_two_stage","random","v1","ENFORCEMENT")),
    ("Q4","explicit ENF vs noisy ENF","uB_minus_uA",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","ENFORCEMENT"),
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","NOISY_ENFORCEMENT")),
    ("Q5","explicit ENF vs costly ENF","uB_minus_uA",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","ENFORCEMENT"),
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","COSTLY_ENFORCEMENT")),
    ("Q6","explicit TEMPT vs explicit ENF","uB_minus_uA",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","TEMPTATION"),
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","ENFORCEMENT")),
    ("Q7","explicit NORMAL vs explicit TEMPT","uB_minus_uA",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","NORMAL"),
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","TEMPTATION")),
    ("Q8","explicit ENF vs light ENF on valid reports","valid_reports",
     filt("qwen3:8b","resource_two_stage","explicit_roles","v3","ENFORCEMENT"),
     filt("qwen3:8b","resource_two_stage","light_roles","v3","ENFORCEMENT")),
]

test_rows = []
for tid, label, metric, A, B in tests:
    a = vals(A, metric)
    b = vals(B, metric)
    test_rows.append({
        "id": tid,
        "comparison": label,
        "metric": metric,
        "mean_A": round(mean(a), 4),
        "mean_B": round(mean(b), 4),
        "diff_A_minus_B": round(mean(a)-mean(b), 4),
        "cohen_d": round(cohend(a, b), 4),
        "permutation_p": round(perm_p(a[:], b[:], n=10000, seed=0), 6),
        "n_A": len(a),
        "n_B": len(b),
    })

# 6. Representative traces
def best_trace(model, env, role, prompt, cond, target_metric, choose="max"):
    sub = filt(model=model, env=env, role=role, prompt=prompt, cond=cond)
    if choose == "max":
        sub = sorted(sub, key=lambda x: x[target_metric], reverse=True)
    else:
        sub = sorted(sub, key=lambda x: x[target_metric])
    return sub[0] if sub else None

trace_specs = [
    ("qwen_resource_explicit_v3_temptation", "qwen3:8b", "resource_two_stage", "explicit_roles", "v3", "TEMPTATION", "uB_minus_uA", "max"),
    ("qwen_resource_explicit_v3_enforcement", "qwen3:8b", "resource_two_stage", "explicit_roles", "v3", "ENFORCEMENT", "valid_reports", "max"),
    ("qwen_resource_explicit_v3_noisy", "qwen3:8b", "resource_two_stage", "explicit_roles", "v3", "NOISY_ENFORCEMENT", "uB_minus_uA", "max"),
    ("qwen_turn_explicit_v3_enforcement", "qwen3:8b", "turn_taking_two_stage", "explicit_roles", "v3", "ENFORCEMENT", "valid_reports", "max"),
]
trace_index = []
for name, model, env, role, prompt, cond, metric, choose in trace_specs:
    tr = best_trace(model, env, role, prompt, cond, metric, choose)
    if tr is not None:
        out = OUT / f"{name}.json"
        out.write_text(json.dumps(tr, indent=2), encoding="utf-8")
        trace_index.append({
            "trace_name": name,
            "model": model,
            "environment": env,
            "role_mode": role,
            "prompt_variant": prompt,
            "condition": cond,
            "seed": tr["seed"],
            "path": str(out)
        })

# write all outputs
write_csv(OUT / "table_main_qwen_resource_v3.csv", main_rows)
write_md_table(OUT / "table_main_qwen_resource_v3.md", main_rows)

write_csv(OUT / "table_prompt_sensitivity_qwen_resource.csv", prompt_rows)
write_md_table(OUT / "table_prompt_sensitivity_qwen_resource.md", prompt_rows)

write_csv(OUT / "table_second_norm_qwen_turn_taking.csv", turn_rows)
write_md_table(OUT / "table_second_norm_qwen_turn_taking.md", turn_rows)

write_csv(OUT / "table_llama_robustness.csv", llama_rows)
write_md_table(OUT / "table_llama_robustness.md", llama_rows)

write_csv(OUT / "table_pairwise_tests.csv", test_rows)
write_md_table(OUT / "table_pairwise_tests.md", test_rows)

write_csv(OUT / "trace_index.csv", trace_index)
write_md_table(OUT / "trace_index.md", trace_index)

summary = {
    "frozen_run_file": str(RUNS),
    "main_table_rows": len(main_rows),
    "prompt_table_rows": len(prompt_rows),
    "turn_table_rows": len(turn_rows),
    "llama_table_rows": len(llama_rows),
    "pairwise_test_rows": len(test_rows),
    "trace_count": len(trace_index),
}
(OUT / "analysis_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Saved final analysis package to:", OUT)
print(json.dumps(summary, indent=2))
