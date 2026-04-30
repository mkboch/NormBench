import os
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".")
FIGDIR = ROOT / "paper_assets" / "figures"
TABDIR = ROOT / "paper_assets" / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Global font sizing: enlarged
# ----------------------------
BASE = 20
plt.rcParams.update({
    "font.size": BASE,
    "axes.titlesize": BASE,
    "axes.labelsize": BASE,
    "xtick.labelsize": BASE - 2,
    "ytick.labelsize": BASE - 2,
    "legend.fontsize": BASE - 2,
    "figure.titlesize": BASE,
})

def save_bar_plot(filename, categories, values, ylabel, title="", errors=None, rotation=0, width=0.65):
    plt.figure(figsize=(12, 8))
    x = np.arange(len(categories))
    if errors is None:
        plt.bar(x, values, width=width)
    else:
        plt.bar(x, values, yerr=errors, capsize=6, width=width)
    plt.xticks(x, categories, rotation=rotation)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGDIR / filename, dpi=220, bbox_inches="tight")
    plt.close()

def save_grouped_bar_plot(filename, categories, series_names, series_values, ylabel, title="", errors=None):
    plt.figure(figsize=(13, 8))
    x = np.arange(len(categories))
    n = len(series_names)
    total_width = 0.8
    bar_width = total_width / n
    offsets = (np.arange(n) - (n - 1) / 2) * bar_width

    for i, name in enumerate(series_names):
        vals = series_values[i]
        errs = None if errors is None else errors[i]
        if errs is None:
            plt.bar(x + offsets[i], vals, width=bar_width, label=name)
        else:
            plt.bar(x + offsets[i], vals, yerr=errs, capsize=5, width=bar_width, label=name)

    plt.xticks(x, categories)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGDIR / filename, dpi=220, bbox_inches="tight")
    plt.close()

# ---------------------------------------
# Frozen values from the v06 final package
# ---------------------------------------

conditions = ["Normal", "Temptation", "Enforcement", "Costly\nEnforcement", "Noisy\nEnforcement"]

# Qwen resource explicit_roles v3
qwen_resource_explicit_v3 = {
    "uB_minus_uA": [5.0000, 15.2000, -4.7667, -3.1667, 8.0000],
    "uB_minus_uA_sd": [0.0000, 0.7483, 0.4230, 2.4233, 5.1833],
    "over_take": [5.0000, 5.0000, 5.0000, 5.0000, 5.0000],
    "over_take_sd": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    "valid_reports": [0.0000, 0.0000, 5.0000, 4.9667, 1.7667],
    "valid_reports_sd": [0.0000, 0.0000, 0.0000, 0.1795, 1.2828],
}

# Role ablation: resource environment, v3, Qwen
role_modes = ["Explicit", "Light", "Uniform", "Random"]
role_ablation_enf = {
    "uB_minus_uA": [-4.7667, 1.7667, 0.2667, 0.4000],
    "uB_minus_uA_sd": [0.4230, 2.0766, 0.9978, 3.9967],
    "valid_reports": [5.0000, 0.0333, 0.0667, 0.9333],
    "valid_reports_sd": [0.0000, 0.1795, 0.2494, 0.8138],
    "precision": [1.0000, 0.0333, 0.0667, 0.6333],
    "precision_sd": [0.0000, 0.1795, 0.2494, 0.4819],
}

# Prompt sensitivity: Qwen resource explicit_roles
prompt_variants = ["v1", "v2", "v3"]
prompt_cond = ["Normal", "Temptation", "Enforcement", "Costly", "Noisy"]
prompt_uB = np.array([
    [5.0000, 20.0000, -4.1667, 14.0000, 18.1333],
    [5.0000, 17.8667, -4.7333, 2.0667, 14.2667],
    [5.0000, 15.2000, -4.7667, -3.1667, 8.0000],
])
prompt_valid = np.array([
    [0.0000, 0.0000, 4.9333, 1.5000, 0.4667],
    [0.0000, 0.0000, 4.9333, 3.7000, 0.4667],
    [0.0000, 0.0000, 5.0000, 4.9667, 1.7667],
])

# Qwen turn-taking explicit_roles v3
qwen_turn_explicit_v3 = {
    "uB_minus_uA": [3.1333, 7.6000, -2.4333, -3.1667, 3.6667],
    "uB_minus_uA_sd": [0.4989, 0.9165, 3.1057, 0.8596, 4.0277],
    "valid_reports": [0.0000, 0.0000, 3.4333, 3.6667, 2.2333],
    "valid_reports_sd": [0.0000, 0.0000, 0.4955, 0.6498, 0.9195],
    "violation_rate": [1.9667, 1.1000, 0.8267, 0.7533, 1.0467],
    "violation_rate_sd": [0.1043, 0.1342, 0.2407, 0.1231, 0.2667],
}

# Llama resource explicit_roles v3
llama_resource_explicit_v3 = {
    "uB_minus_uA": [6.0667, 16.3667, 1.1000, 2.1667, 5.8333],
    "uB_minus_uA_sd": [2.6575, 5.1087, 1.6197, 2.7699, 3.9672],
    "over_take": [2.6000, 4.0000, 2.7000, 3.5000, 3.4667],
    "over_take_sd": [1.6452, 1.4142, 1.7916, 1.9451, 1.6069],
    "valid_reports": [0.0000, 0.0000, 2.5667, 3.0667, 2.1333],
    "valid_reports_sd": [0.0000, 0.0000, 1.8200, 1.8962, 1.3840],
}

# --------------------------------
# Generate enlarged figure PNG files
# --------------------------------

# fig2 primary qwen
save_bar_plot(
    "fig2a_qwen_uB_minus_uA.png",
    conditions,
    qwen_resource_explicit_v3["uB_minus_uA"],
    ylabel=r"$u_B-u_A$",
    errors=qwen_resource_explicit_v3["uB_minus_uA_sd"],
)
save_bar_plot(
    "fig2b_qwen_over_take.png",
    conditions,
    qwen_resource_explicit_v3["over_take"],
    ylabel="Over-take violations",
    errors=qwen_resource_explicit_v3["over_take_sd"],
)
save_bar_plot(
    "fig2c_qwen_valid_reports.png",
    conditions,
    qwen_resource_explicit_v3["valid_reports"],
    ylabel="Valid reports",
    errors=qwen_resource_explicit_v3["valid_reports_sd"],
)

# fig3 role ablation
save_bar_plot(
    "fig3a_role_ablation_uB_minus_uA.png",
    role_modes,
    role_ablation_enf["uB_minus_uA"],
    ylabel=r"$u_B-u_A$",
    errors=role_ablation_enf["uB_minus_uA_sd"],
)
save_bar_plot(
    "fig3b_role_ablation_valid_reports.png",
    role_modes,
    role_ablation_enf["valid_reports"],
    ylabel="Valid reports",
    errors=role_ablation_enf["valid_reports_sd"],
)
save_bar_plot(
    "fig3c_role_ablation_precision.png",
    role_modes,
    role_ablation_enf["precision"],
    ylabel="Report precision",
    errors=role_ablation_enf["precision_sd"],
)

# fig4 prompt sensitivity
save_grouped_bar_plot(
    "fig4_prompt_sensitivity_qwen_resource.png",
    prompt_cond,
    prompt_variants,
    prompt_uB,
    ylabel=r"$u_B-u_A$",
)

# fig5 turn taking qwen
save_bar_plot(
    "fig5a_turn_taking_uB_minus_uA.png",
    conditions,
    qwen_turn_explicit_v3["uB_minus_uA"],
    ylabel=r"$u_B-u_A$",
    errors=qwen_turn_explicit_v3["uB_minus_uA_sd"],
)
save_bar_plot(
    "fig5b_turn_taking_valid_reports.png",
    conditions,
    qwen_turn_explicit_v3["valid_reports"],
    ylabel="Valid reports",
    errors=qwen_turn_explicit_v3["valid_reports_sd"],
)
save_bar_plot(
    "fig5c_turn_taking_violation_rate.png",
    conditions,
    qwen_turn_explicit_v3["violation_rate"],
    ylabel="Violation rate per round",
    errors=qwen_turn_explicit_v3["violation_rate_sd"],
)

# fig6 llama
save_bar_plot(
    "fig6a_llama_uB_minus_uA.png",
    conditions,
    llama_resource_explicit_v3["uB_minus_uA"],
    ylabel=r"$u_B-u_A$",
    errors=llama_resource_explicit_v3["uB_minus_uA_sd"],
)
save_bar_plot(
    "fig6b_llama_over_take.png",
    conditions,
    llama_resource_explicit_v3["over_take"],
    ylabel="Over-take violations",
    errors=llama_resource_explicit_v3["over_take_sd"],
)
save_bar_plot(
    "fig6c_llama_valid_reports.png",
    conditions,
    llama_resource_explicit_v3["valid_reports"],
    ylabel="Valid reports",
    errors=llama_resource_explicit_v3["valid_reports_sd"],
)

# --------------------------------
# Write compact table tex files too
# --------------------------------

tab1 = r"""
\begin{tabular}{lcccc}
\toprule
Condition & $u_B-u_A$ & Over-take & Valid reports & Report precision \\
\midrule
NORMAL & 5.000 (0.000) & 5.000 (0.000) & 0.000 (0.000) & 0.000 (0.000) \\
TEMPTATION & 15.200 (0.748) & 5.000 (0.000) & 0.000 (0.000) & 0.000 (0.000) \\
ENFORCEMENT & -4.767 (0.423) & 5.000 (0.000) & 5.000 (0.000) & 1.000 (0.000) \\
COSTLY ENFORCEMENT & -3.167 (2.423) & 5.000 (0.000) & 4.967 (0.179) & 1.000 (0.000) \\
NOISY ENFORCEMENT & 8.000 (5.183) & 5.000 (0.000) & 1.767 (1.283) & 0.833 (0.373) \\
\bottomrule
\end{tabular}
""".strip()

tab2 = r"""
\begin{tabular}{p{4.8cm}ccc}
\toprule
Comparison & Metric & Mean diff & $p$ \\
\midrule
Explicit ENF vs Light ENF & $u_B-u_A$ & -6.533 & 0.0001 \\
Explicit ENF vs Uniform ENF & $u_B-u_A$ & -5.033 & 0.0001 \\
Explicit ENF vs Random ENF & $u_B-u_A$ & -5.167 & 0.0001 \\
Explicit ENF vs Noisy ENF & $u_B-u_A$ & -12.767 & 0.0001 \\
Explicit ENF vs Costly ENF & $u_B-u_A$ & -1.600 & 0.0006 \\
Explicit TEMPTATION vs Explicit ENF & $u_B-u_A$ & 19.967 & 0.0001 \\
Explicit NORMAL vs Explicit TEMPTATION & $u_B-u_A$ & -10.200 & 0.0001 \\
Explicit ENF vs Light ENF & valid reports & 4.967 & 0.0001 \\
\bottomrule
\end{tabular}
""".strip()

tab3 = r"""
\begin{tabular}{p{3.8cm}ccc}
\toprule
Setting & TEMPTATION $u_B-u_A$ & ENFORCEMENT $u_B-u_A$ & ENFORCEMENT valid reports \\
\midrule
Qwen / resource & 15.200 (0.748) & -4.767 (0.423) & 5.000 (0.000) \\
Qwen / turn-taking & 7.600 (0.916) & -2.433 (3.106) & 3.433 (0.495) \\
Llama / resource & 16.367 (5.109) & 1.100 (1.620) & 2.567 (1.820) \\
\bottomrule
\end{tabular}
""".strip()

(TABDIR / "tab1_main_qwen_resource_v3.tex").write_text(tab1, encoding="utf-8")
(TABDIR / "tab2_pairwise_tests.tex").write_text(tab2, encoding="utf-8")
(TABDIR / "tab3_transfer_robustness.tex").write_text(tab3, encoding="utf-8")

print("Regenerated enlarged figures in:", FIGDIR)
print("Wrote compact tables in:", TABDIR)
