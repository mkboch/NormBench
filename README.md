# NormBench

**NormBench: A Two-Stage Benchmark for Measuring Compliance and Regulatory Behavior in Local LLM Societies**

NormBench is a controlled diagnostic benchmark for measuring compliance, exploitation, reporting, and audit-induced payoff consequences in local large language model societies. Agents first act in a shared environment and then enter a separate audit stage, allowing reporting decisions to be evaluated against current-round events.

The benchmark is intentionally minimal. Its purpose is not to model spontaneous norm emergence or learned social adaptation, but to isolate how role specification, incentives, audit settings, and prompt formulation affect measurable regulatory behavior.

This repository contains the benchmark implementation, released results, analysis scripts and artifacts, and paper-ready figures and tables.

## Benchmark design

NormBench uses a two-stage per-round interaction protocol:

1. **Action stage:** agents choose actions in a shared environment.
2. **Audit stage:** current-round event records enter the audit stage under the condition-specific reporting rule, and the monitor may report an eligible violator or abstain.
3. **Utility update:** reporting costs, valid-report rewards, and violator penalties are applied.
4. **Repeat:** the process continues until the maximum five-round horizon is reached or the shared token budget is exhausted.

Each episode begins with a shared budget of `18` tokens.

For non-random LLM conditions, agents choose either `take` or `wait`. The requested amount is an integer in `{0, 1, 2}`; `wait` forces the amount to `0`, and realized takes are capped by the remaining token budget.

A report is valid only when its target is an eligible current-round violator. Each submitted report first incurs any condition-specific reporting cost. A valid report then gives the monitor `+2` utility and applies a fixed `-4` penalty to the reported violator. False reports receive no valid-report reward or violator penalty, although they still incur any applicable reporting cost.

The two-stage design ensures that current-round evidence precedes the reporting decision and makes report validity directly checkable. It does not establish that a report is causally driven by the evidence rather than by prompt wording; prompt dependence is evaluated separately through controlled prompt variants.

## Environments

NormBench includes two norm families:

- `resource_two_stage`: agents interact through a shared token pool. Taking more than one token in a round is an over-take violation.
- `turn_taking_two_stage`: the designated agent cycles through agents A, B, and C. Any positive take by a non-designated agent is an out-of-turn violation.

The resource environment is used for detailed analysis, while the turn-taking environment provides a second norm family for qualitative transfer analysis.

The numerical `+2` temptation bonus applies to over-take violations in the resource environment. The turn-taking environment does not apply this numerical temptation bonus to out-of-turn violations, so transfer between the two environments should be interpreted qualitatively rather than as a matched incentive replication.

## Role modes

The benchmark evaluates four role modes:

- `explicit_roles`: a compliant cooperator, an opportunistic maximizer, and a monitor/enforcer.
- `light_roles`: weaker role labels with less direct behavioral scripting.
- `uniform`: all agents receive the same generic payoff-seeking instruction.
- `random`: a hand-coded non-strategic reference rather than an LLM-prompted role condition.

For the random reference, actions are sampled from the environment-specific action space. When reporting is enabled, the baseline reports a randomly selected eligible violator with probability `0.5` and otherwise abstains.

## Conditions

The benchmark evaluates five conditions:

| Condition | Resource temptation bonus | Reporting enabled | Report cost | Report-eligible probability |
|---|---:|:---:|---:|---:|
| `NORMAL` | 0 | No | 0 | 1.0 |
| `TEMPTATION` | 2 | No | 0 | 1.0 |
| `ENFORCEMENT` | 2 | Yes | 0 | 1.0 |
| `COSTLY_ENFORCEMENT` | 2 | Yes | 1 | 1.0 |
| `NOISY_ENFORCEMENT` | 2 | Yes | 1 | 0.6 |

Under `NOISY_ENFORCEMENT`, each current-round event independently receives a report-eligibility flag with probability `0.6`. Only eligible violations count as valid report targets. Event records remain present in the audit prompt together with their eligibility flags.

## Protocol settings

The released paper configuration uses:

- models: `qwen3:8b`, `llama3.1:8b`
- prompt variants for LLM settings: `v1`, `v2`, `v3`
- random reference: prompt-free, stored under `v1` for bookkeeping
- seeds: `30` per configuration
- maximum rounds: `5` per episode
- shared starting token budget: `18`
- early termination: when the token budget is exhausted
- temperature: `0.3`
- top-p: `0.9`
- generation limit: `512` tokens
- context window: `4096` tokens
- JSON-constrained output
- reasoning disabled

## Metrics

NormBench reports the following metrics:

- violation count
- valid reports
- report precision
- payoff gap \(u_B - u_A\)
- violation rate per round

Report precision is defined as the number of valid reports divided by the number of submitted reports and is set to `0` when no report is submitted.

Violation rate per round is the total number of recorded violations divided by the number of completed rounds.

The payoff gap \(u_B - u_A\) compares the final utilities of agents B and A. Under explicit roles, this corresponds to the opportunist-minus-cooperator payoff gap.

Together, these metrics distinguish violation behavior, oversight quality, and whether audit changes the payoff consequences of opportunistic behavior.

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
- `normbench_v06_human_template_1777118963.csv`: human-readable template included in the release.
- `frontier_eval_scaffold_1777118963.json`: evaluation scaffold included in the release package.

### `final_analysis/`

- `build_final_analysis_v06.py`: builds the final analysis artifacts.
- `table_main_qwen_resource_v3.*`: detailed Qwen resource results.
- `table_pairwise_tests.*`: pairwise comparison results.
- `table_prompt_sensitivity_qwen_resource.*`: prompt-sensitivity results.
- `table_second_norm_qwen_turn_taking.*`: transfer results for the turn-taking norm family.
- `table_llama_robustness.*`: second-model robustness results.
- `trace_index.*`: index of representative traces.

### `paper_assets/`

- `figures/`: paper figures.
- `tables/`: paper tables in LaTeX format.

## Key reported findings

Across the explicit-role resource setting, the core Temptation-to-Enforcement contrast is directionally consistent across all six tested model-prompt combinations: two models multiplied by three prompt variants. Enforcement reduces the opportunist payoff gap relative to Temptation in all six combinations, with reductions ranging from `15.27` to `24.17` payoff units.

Complete reversal to a negative Enforcement payoff gap occurs in all three Qwen3:8B prompt variants but in none of the three Llama 3.1:8B variants. The direction of the core contrast is therefore stable across the tested prompts and models, while the magnitude and complete reversal of opportunistic advantage remain model-dependent.

For detailed presentation, the paper reports `qwen3:8b` in `resource_two_stage` under `explicit_roles` and prompt variant `v3`. In this setting:

- `NORMAL`: \(u_B-u_A = 5.0000\)
- `TEMPTATION`: \(u_B-u_A = 15.2000\)
- `ENFORCEMENT`: \(u_B-u_A = -4.7667\), with `5.0000` valid reports and perfect report precision
- `COSTLY_ENFORCEMENT`: \(u_B-u_A = -3.1667\)
- `NOISY_ENFORCEMENT`: \(u_B-u_A = 8.0000\)

The mean over-take count remains `5.0000` across these five detailed Qwen conditions. The Enforcement effect is therefore monitor-driven: audit and reporting change the payoff consequences of repeated violations rather than inducing the opportunistic agent to become compliant.

The role ablation further shows that the strongest regulatory response depends heavily on explicit role specification. This is treated as a construct-validity boundary rather than evidence of spontaneous norm emergence.

Prompt sensitivity is also condition-dependent. Core Enforcement remains relatively stable across the three prompt variants, while `COSTLY_ENFORCEMENT` and `NOISY_ENFORCEMENT` vary substantially more.

The turn-taking environment provides a qualitative second-norm-family transfer test. Because it does not use the same numerical temptation bonus as the resource environment, its results should not be interpreted as a matched incentive replication.

## Reproducing the released artifacts

To launch the released benchmark pipeline:

```bash
bash source/run_full_v06_detached.sh
```

To run the main implementation directly, adjust local paths and model-serving configuration as needed:

```bash
python source/normbench_v06.py
```

To print the latest released summary:

```bash
python source/print_latest_summary.py
```

The released results already included in this repository are sufficient for inspection and downstream analysis without rerunning the models.

## Data and code availability

All reported results are derived from the released benchmark outputs and analysis artifacts in this repository.

The repository is intended to support inspection, reproduction, and extension of the NormBench workflow. Because the benchmark is deliberately compact and evaluates only two local 8B-scale models, three-agent societies, and a maximum five-round horizon, the released results should not be interpreted as model-universal evidence about LLM societies or as evidence of spontaneous norm emergence or long-horizon social adaptation.
