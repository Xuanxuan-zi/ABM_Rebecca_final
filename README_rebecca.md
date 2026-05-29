# International Students Cultural Segregation ABM

This repository contains an agent-based model of cultural segregation and cross-cultural friendship formation among international students on a university campus. The model explores how small interaction costs, cultural distance, English fluency, stress, local movement, and repeated encounters can produce larger patterns of spatial clustering and network homophily over time. It also tests whether a university-supported structured pairing intervention can increase cross-cultural friendship.

## Research Question

How can small interaction costs between culturally different students produce larger patterns of social segregation, and can a university-supported structured pairing intervention increase cross-cultural friendship?

## Model Overview

The model represents a university campus as a grid. Each student is an agent who moves around the campus, interacts with nearby students, experiences comfort or stress, and may form or lose maintained social ties over time.

Each student has the following key attributes:

- Country group
- Cultural vector
- English fluency
- Stress level
- Isolation status
- Maintained social ties
- Encounter history with other students

The model focuses on how repeated local interactions can gradually produce aggregate social patterns. Students do not begin with fixed friendship networks. Instead, friendships emerge through repeated encounters, cultural similarity, language fluency, and the structured pairing intervention.

## Main Social Mechanisms

The model contains five main rules.

### R1: Similarity-Based Social Tie Formation

When two students have low cultural distance, their interaction is relatively easy. These interactions reduce stress and can create or strengthen a maintained social tie after repeated encounters.

### R2: Cross-Cultural Learning

When two students have moderate cultural distance, they may learn from each other. Their cultural vectors move slightly closer, and cross-country interaction can improve English fluency. However, these interactions also create a small stress cost because learning across difference requires effort.

### R3: Conflict and Isolation

When two students have high cultural distance, their interaction can increase stress. Repeated stressful encounters can create conflict memory. If a student's stress becomes too high, the student temporarily withdraws from campus interaction and becomes isolated.

### R4: Movement and Spatial Clustering

Students move through the campus grid and tend to prefer local areas with more culturally similar neighbors. This rule allows small individual preferences for familiar social spaces to produce larger patterns of spatial clustering.

### R5: Structured Pairing Intervention

The university-supported intervention pairs students from different cultural groups. The intervention creates additional structured encounters between students who may not otherwise interact. These encounters can reduce stress, support learning, and increase the probability of cross-cultural social tie formation.

## Population Scenarios

The model compares three population scenarios.

| Scenario | US | China | India | South Korea | Germany | Brazil |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 50% | 15% | 12% | 8% | 8% | 7% |
| Chinese-heavy | 30% | 50% | 5% | 5% | 5% | 5% |
| Balanced | 25% | 25% | 25% | 10% | 10% | 5% |

Each scenario is run both with and without the R5 structured pairing intervention.

## Main Outcomes

The model tracks several outcomes:

- `spatial_clustering`
- `network_homophily`
- `cross_cultural_friendship_share`
- `mean_stress`
- `mean_stress_active`
- `isolation_rate`
- `active_social_ties`
- `mean_tie_strength`
- `mean_cultural_dist`

These outcomes are used to compare whether the intervention changes network integration, spatial clustering, stress, isolation, and tie formation.

## Repository Structure

```text
.
├── agents.py
├── model.py
├── app.py
├── run_batch.py
├── visualize_batch_results.py
├── batch_results_all_ticks.csv
├── batch_results_final_tick.csv
├── batch_summary_mean.csv
├── batch_summary_mean_sd.csv
├── figures/
│   └── gui_screenshot.png
└── README.md
```

## File Descriptions

### `agents.py`

This file defines the student agent. It includes the cultural vector, English fluency, stress level, isolation status, repeated encounter history, movement behavior, and interaction rules. It also contains the default model parameters for R1 to R5.

### `model.py`

This file defines the main campus model. It initializes the student population, creates the grid, manages social ties, applies the structured pairing intervention, collects model outcomes, and contains the batch experiment function.

### `app.py`

This file runs the interactive Solara visualization. It allows the user to change the number of agents, random seed, population scenario, and intervention setting. It also visualizes the campus grid and plots the model outcomes over time.

### `run_batch.py`

This file runs the batch experiments. It compares the three population scenarios with and without the structured pairing intervention. The current setting uses 20 repeated runs per condition, 300 ticks per run, and 500 agents.

### `visualize_batch_results.py`

This file creates polished figures and summary tables from the batch experiment results. It uses the final-tick batch results to compare outcomes across scenarios and intervention conditions.

### CSV Output Files

The batch experiment produces four CSV files:

- `batch_results_all_ticks.csv`  
  Contains all model outcomes for every tick of every run.

- `batch_results_final_tick.csv`  
  Contains only the final tick from each run.

- `batch_summary_mean.csv`  
  Contains mean outcomes by scenario and intervention condition.

- `batch_summary_mean_sd.csv`  
  Contains mean and standard deviation outcomes by scenario and intervention condition.

## Requirements

This project uses Python 3 and the following packages:

```bash
mesa
solara
numpy
pandas
matplotlib
```

To install the required packages, run:

```bash
pip install mesa solara numpy pandas matplotlib
```

The model was developed using Mesa 3.5.0.

## How to Run the Interactive Model

To run the interactive Solara visualization, use:

```bash
solara run app.py
```

This opens a browser-based model interface. The user can adjust the random seed, number of agents, population scenario, and intervention setting. The interface displays the campus space and several outcome plots.

## Interactive GUI Screenshot

The screenshot below shows the Solara-based GUI after the model is running. The interface allows the user to change the random seed, number of agents, population scenario, and R5 structured pairing intervention setting. It also displays the campus grid and tracks model outcomes over time.

![Interactive GUI Screenshot](https://raw.githubusercontent.com/Xuanxuan-zi/ABM_Rebecca_final/main/OneScreenshot_inteface.png)
## How to Run Batch Experiments

To run the batch experiments, use:

```bash
python3 run_batch.py
```

This runs all combinations of population scenario and intervention condition. The script saves the following output files:

```text
batch_results_all_ticks.csv
batch_results_final_tick.csv
batch_summary_mean.csv
batch_summary_mean_sd.csv
```

The current batch experiment settings are:

```text
Number of runs per condition: 20
Number of ticks per run: 300
Number of agents: 500
Grid size: 50 x 50
Population scenarios: baseline, chinese_heavy, balanced
Intervention conditions: without intervention, with intervention
```

## How to Generate Figures

To generate figures from the batch results, use:

```bash
python3 visualize_batch_results.py
```

This reads `batch_results_final_tick.csv` and creates a folder called `figures_fancy/`. The folder contains figure files and additional summary tables.

The expected outputs include:

```text
figures_fancy/fig_01_main_network_outcomes.png
figures_fancy/fig_02_policy_effect_lollipop.png
figures_fancy/fig_03_spatial_and_welfare_outcomes.png
figures_fancy/fig_04_full_outcome_dashboard.png
figures_fancy/fancy_summary_table.csv
figures_fancy/fancy_intervention_effects.csv
```

## Interpretation of the Model

The model does not claim to provide an empirical estimate of real international student friendships. Instead, it is a theoretical and exploratory agent-based model. It uses stylized assumptions to examine how micro-level interaction rules can generate macro-level patterns.

The main theoretical idea is that social segregation does not have to come only from explicit exclusion. It can also emerge from repeated small differences in comfort, language fluency, stress, and local movement. Over time, these small mechanisms can produce durable patterns of network homophily and spatial clustering.

The structured pairing intervention is designed to test whether institutional support can increase cross-cultural contact. The intervention mainly affects network-level outcomes by creating more opportunities for cross-cultural friendship. However, spatial clustering may not decline as much because movement is still shaped by local comfort, stress, and familiar social surroundings.

## Reproducibility

The batch experiments use random seeds so that results can be reproduced. The base seed is set in `run_batch.py`. Each scenario and intervention condition is repeated across multiple runs to reduce dependence on one random simulation.

To reproduce the main results:

1. Run the batch experiment:

```bash
python3 run_batch.py
```

2. Generate the figures:

```bash
python3 visualize_batch_results.py
```

3. Use the CSV files and generated figures for analysis and reporting.

## External Resources Statement

In completing this project, I used several external resources for technical support, code readability, and documentation. I consulted course materials and examples from class to understand the general structure of an agent-based model, including how to define agents, model rules, step functions, data collection, batch runs, and visualization. I also referred to the official Mesa and Solara documentation when working on the model structure and the interactive visualization interface.

I used generative AI tools as a coding and writing assistant. Specifically, I used AI to help polish code comments, improve the organization of the code, make variable names and section headings clearer, and make the code easier to read. I also used AI to help debug small syntax or structure issues and to improve the wording of explanations in the README. However, I reviewed and modified the suggestions myself. The research question, model logic, agent rules, intervention design, scenario choices, batch experiment structure, and interpretation of results are my own.

I did not use AI to replace the original design or implementation of the project. I did not copy an external model or use external code without review. All final files in this repository were checked and organized by me, and I am responsible for the final model, code, results, and written interpretation.
