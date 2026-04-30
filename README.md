# NormBench

NormBench is a benchmark for measuring compliance, exploitation, reporting, and payoff suppression in local large language model societies. It evaluates how models behave when they act in a shared environment first and only later enter an audit stage where current-round actions become visible and reporting is allowed [web:1].

The repository contains the implementation used to run the benchmark, the released results, the analysis scripts and artifacts, and the paper-ready figures and tables [web:1].

## Benchmark design

NormBench uses a two-stage interaction protocol:

1. **Action stage**: agents choose actions in a shared environment.
2. **Audit stage**: current-round behavior becomes visible, and a monitor may report violations.
3. **Update stage**: rewards and penalties are applied.
4. **Repeat**: the process continues for the full episode.

This design separates action from oversight, which makes reporting behavior directly measurable [web:1].

## Environments

The benchmark includes two environments:

- `resource_two_stage`: a shared-resource setting in which taking more than one token in a round is an over-take violation.
- `turn_taking_two_stage`: a turn-taking setting in which acting outside the designated order is an out-of-turn violation.

## Role modes

The benchmark evaluates four role modes:

- `explicit_roles`
- `light_roles`
- `uniform`
- `random`

## Conditions

The benchmark evaluates five conditions:

- `NORMAL`
- `TEMPTATION`
- `ENFORCEMENT`
- `COSTLY_ENFORCEMENT`
- `NOISY_ENFORCEMENT`

## Protocol settings

The released paper configuration uses:

- models: `qwen3:8b`, `llama3.1:8b`
- prompt variants: `v1`, `v2`, `v3`
- random baseline prompt variant: `v1`
- seeds: `30` per configuration
- rounds: `5` per episode

## Metrics

NormBench reports the following metrics:

- violations
- valid reports
- report precision
- payoff gap \(u_B - u_A\)
- violation rate per round

These metrics distinguish between basic compliance, opportunistic exploitation, and audit-enabled regulation.

## Repository layout

```text
NormBench/
├── README.md
├── source/
│   ├── normbench_v06.py
│   ├── print_latest_summary.py
│   └── run_full_v06_detached.sh
├── results/
│   ├── frontier_eval_scaffold_1777118963.json
│   ├── normbench_v06_human_template_1777118963.csv
│   ├── normbench_v06_summary_1777118963.csv
│   └── normbench_v06_summary_1777118963.json
├── final_analysis/
│   ├── analysis_manifest.json
│   ├── build_final_analysis_v06.py
│   ├── table_main_qwen_resource_v3.csv
│   ├── table_main_qwen_resource_v3.md
│   ├── table_pairwise_tests.csv
│   ├── table_pairwise_tests.md
│   ├── table_prompt_sensitivity_qwen_resource.csv
│   ├── table_prompt_sensitivity_qwen_resource.md
│   ├── table_second_norm_qwen_turn_taking.csv
│   ├── table_second_norm_qwen_turn_taking.md
│   ├── table_llama_robustness.csv
│   ├── table_llama_robustness.md
│   ├── trace_index.csv
│   ├── trace_index.md
│   └── representative trace files
├── paper_assets/
│   ├── figures/
│   └── tables/
```

## Key files

### `source/`
- `normbench_v06.py`: main benchmark implementation.
- `print_latest_summary.py`: prints the most recent released summary outputs.
- `run_full_v06_detached.sh`: launches the benchmark pipeline in detached mode.

### `results/`
- `normbench_v06_summary_1777118963.csv` and `.json`: released summary outputs.
- `normbench_v06_human_template_1777118963.csv`: human-readable template used in the release.
- `frontier_eval_scaffold_1777118963.json`: evaluation scaffold used in the release package.

### `final_analysis/`
- `build_final_analysis_v06.py`: builds the final analysis artifacts.
- `table_main_qwen_resource_v3.*`: main results for the Qwen resource benchmark.
- `table_pairwise_tests.*`: pairwise comparison results.
- `table_prompt_sensitivity_qwen_resource.*`: prompt sensitivity results.
- `table_second_norm_qwen_turn_taking.*`: transfer results for the turn-taking norm family.
- `table_llama_robustness.*`: robustness results for Llama.
- `trace_index.*`: index of representative traces.

### `paper_assets/`
- `figures/`: paper figures.
- `tables/`: paper tables in LaTeX format.

## Main paper result

The main analysis focuses on `qwen3:8b` in `resource_two_stage` under `explicit_roles` and prompt variant `v3`. In that setting, the payoff gap \(u_B - u_A\) rises from `5.0000` in `NORMAL` to `15.2000` in `TEMPTATION`, then falls to `-4.7667` in `ENFORCEMENT`, with `5.0000` valid reports and perfect report precision.

The effect weakens under `COSTLY_ENFORCEMENT` and degrades further under `NOISY_ENFORCEMENT`. The strongest regulatory effect depends on explicit role scripting, and stress conditions are more sensitive to prompt choice than core enforcement.

## Reproducing the released artifacts

To rebuild the benchmark pipeline:

```bash
bash source/run_full_v06_detached.sh
```

To run the main implementation directly, adjust paths as needed:

```bash
python source/normbench_v06.py
```

To print the latest released summary:

```bash
python source/print_latest_summary.py
```

The released results already included in this repository are sufficient for inspection, analysis, and citation.

## Data and code availability

All reported results are derived from the released benchmark outputs and the analysis artifacts in this repository. The repository is intended to support inspection, reproduction, and extension of the benchmark workflow.
