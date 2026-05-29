import numpy as np
import networkx as nx
from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from agents import Student, DEFAULT_PARAMS


## Three stylized campus compositions.
## The smaller groups stand in for representative international student groups,
## not a complete list of all countries on a real campus.
SCENARIOS = {
    "baseline": {
        "US": 0.50,
        "China": 0.15,
        "India": 0.12,
        "S.Korea": 0.08,
        "Germany": 0.08,
        "Brazil": 0.07,
    },
    "chinese_heavy": {
        "US": 0.30,
        "China": 0.50,
        "India": 0.05,
        "S.Korea": 0.05,
        "Germany": 0.05,
        "Brazil": 0.05,
    },
    "balanced": {
        "US": 0.25,
        "China": 0.25,
        "India": 0.25,
        "S.Korea": 0.10,
        "Germany": 0.10,
        "Brazil": 0.05,
    },
}


class CampusModel(Model):
    ## Define model initiation, requiring all major parameter inputs
    def __init__(
        self,
        n_agents=500,
        width=50,
        height=50,
        population_scenario="baseline",
        intervention_enabled=True,
        seed=None,
    ):
        ## Mesa 3.5 uses rng rather than the old seed argument
        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)

        ## Check whether the selected population scenario exists
        if population_scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown population_scenario '{population_scenario}'. "
                f"Choose from {list(SCENARIOS)}."
            )

        if n_agents > width * height:
            raise ValueError("n_agents cannot exceed the number of grid cells")

        ## Store basic model parameters
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.scenario_name = population_scenario
        self.tick = 0
        self.running = True

        ## Copy default parameters so each model instance can modify its own values
        self.params = dict(DEFAULT_PARAMS)
        self.params["r5_enabled"] = intervention_enabled

        ## Calibrated for a 500-agent model:
        ## active_social_ties represent meaningful maintained social ties,
        ## not every casual repeated contact.
        ## This middle setting avoids both extremely high and extremely low tie counts.
        self.params["r1_encounters_to_friend"] = 12
        self.params["r1_tie_strength_increment"] = 0.35

        ## R5 represents a moderately large university program, such as
        ## intercultural orientation groups, buddy programs, or supported mixers.
        ## Each intervention cycle reaches about 20% of the student population.
        self.params["r5_frequency"] = 20
        self.params["r5_coverage_rate"] = 0.20
        self.params["r5_pairs_per_event"] = max(
            1, int(self.n_agents * self.params["r5_coverage_rate"] / 2)
        )
        self.params["r5_min_distance"] = self.params["r1_distance_threshold"]
        self.params["r5_max_distance"] = 0.55
        self.params["r5_min_fluency"] = 0.40
        self.params["r5_encounter_boost"] = 4
        self.params["r5_stress_reduction"] = 0.05
        self.params["r5_learning_rate"] = 0.006
        self.params["r5_language_gain"] = 0.006
        self.params["r5_friendship_threshold"] = 4
        self.params["r5_tie_strength_increment"] = 0.90
        self.params["r5_partner_movement_bonus"] = 1.15

        self.params["tie_decay_rate"] = 0.985
        self.params["tie_removal_threshold"] = 0.25

        ## Create campus grid and maintained social-tie network
        self.grid = SingleGrid(width, height, torus=True)
        self.friendship_network = nx.Graph()

        ## These counters help track friendship dynamics over time
        self.total_ties_created = 0
        self.total_ties_removed = 0

        ## This set prevents one pair from interacting twice in the same tick
        self.interacted_pairs_this_tick = set()

        ## Create and place all agents
        self.create_agents(n_agents, population_scenario)

        ## Define data collector for the main plots and batch outputs
        self.datacollector = DataCollector(
            model_reporters={
                "spatial_clustering": compute_spatial_clustering,
                "network_homophily": compute_network_homophily,
                "cross_cultural_friendship_share": compute_cross_cultural_friendship_share,
                "mean_stress": compute_mean_stress,
                "mean_stress_active": compute_mean_stress_active,
                "isolation_rate": compute_isolation_rate,
                "active_social_ties": lambda m: m.friendship_network.number_of_edges(),
                ## Keep this old name so older notes or plots still make sense
                "friendship_count": lambda m: m.friendship_network.number_of_edges(),
                "mean_tie_strength": compute_mean_tie_strength,
                "ties_created_total": lambda m: m.total_ties_created,
                "ties_removed_total": lambda m: m.total_ties_removed,
                "mean_cultural_dist": compute_mean_cultural_distance,
            }
        )

        ## Initialize data collector before the first model step
        self.datacollector.collect(self)

    ## Create agents according to the chosen scenario composition
    def create_agents(self, n_agents, population_scenario):
        proportions = SCENARIOS[population_scenario]
        countries = list(proportions.keys())
        weights = np.array([proportions[c] for c in countries], dtype=float)
        weights = weights / weights.sum()

        ## Draw exact country counts from a multinomial distribution
        counts = self.rng.multinomial(n_agents, weights)

        ## Randomly choose empty grid cells for agents
        empty_cells = list(self.grid.empties)
        self.rng.shuffle(empty_cells)
        cell_iter = iter(empty_cells)

        for country, count in zip(countries, counts):
            for _ in range(int(count)):
                agent = Student(self, country)
                pos = next(cell_iter)
                self.grid.place_agent(agent, pos)
                self.friendship_network.add_node(agent.unique_id)

    ## Create a new social tie or strengthen an existing one
    def add_or_strengthen_social_tie(self, a, b, amount=1.0):
        if a.unique_id == b.unique_id:
            return

        p = self.params
        amount = float(amount)

        if self.friendship_network.has_edge(a.unique_id, b.unique_id):
            current_weight = self.friendship_network[a.unique_id][b.unique_id].get(
                "weight", 1.0
            )
            new_weight = min(p["tie_max_strength"], current_weight + amount)
            self.friendship_network[a.unique_id][b.unique_id]["weight"] = new_weight
        else:
            self.friendship_network.add_edge(
                a.unique_id,
                b.unique_id,
                weight=min(p["tie_max_strength"], amount),
            )
            self.total_ties_created += 1

    ## Weaken social ties that are not reinforced, and remove very weak ties
    def decay_social_ties(self):
        p = self.params
        edges_to_remove = []

        for u, v, data in list(self.friendship_network.edges(data=True)):
            new_weight = data.get("weight", 1.0) * p["tie_decay_rate"]
            data["weight"] = new_weight

            if new_weight < p["tie_removal_threshold"]:
                edges_to_remove.append((u, v))

        if edges_to_remove:
            self.friendship_network.remove_edges_from(edges_to_remove)
            self.total_ties_removed += len(edges_to_remove)

    ## R5: structured cross-country pairing, like a buddy program
    def r5_pairing_intervention(self):
        p = self.params

        if not p["r5_enabled"]:
            return
        if self.tick == 0 or self.tick % p["r5_frequency"] != 0:
            return

        active_agents = [a for a in self.agents if a.state == "active"]
        if len(active_agents) < 2:
            return

        ## The target is based on unique students, not only pair count.
        ## With 500 agents and 20% coverage, this gives 50 pairs per cycle.
        n_pairs = min(p["r5_pairs_per_event"], len(active_agents) // 2)
        boost = p["r5_encounter_boost"]

        chosen_pairs = set()
        chosen_agents = set()
        attempts = 0
        max_attempts = n_pairs * 100

        while len(chosen_pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            idx = self.rng.choice(len(active_agents), size=2, replace=False)
            a = active_agents[int(idx[0])]
            b = active_agents[int(idx[1])]

            ## Coverage means unique students in each intervention cycle.
            if a.unique_id in chosen_agents or b.unique_id in chosen_agents:
                continue

            pair_key = tuple(sorted((a.unique_id, b.unique_id)))
            if pair_key in chosen_pairs:
                continue

            ## The program intentionally creates cross-country contact.
            if a.country == b.country:
                continue

            distance = a.cultural_distance(b)
            fluency = a.interaction_fluency(b)

            ## Pair students who are different enough to learn from each other,
            ## but not so mismatched that the contact becomes unrealistic.
            if distance <= p["r5_min_distance"]:
                continue
            if distance > p["r5_max_distance"]:
                continue
            if fluency < p["r5_min_fluency"]:
                continue

            chosen_pairs.add(pair_key)
            chosen_agents.add(a.unique_id)
            chosen_agents.add(b.unique_id)

            ## Keep a memory of supported cross-cultural partners.
            a.structured_contact_partners.add(b.unique_id)
            b.structured_contact_partners.add(a.unique_id)

            ## A structured program cycle counts as several meaningful encounters.
            a.encounter_count[b.unique_id] = (
                a.encounter_count.get(b.unique_id, 0) + boost
            )
            b.encounter_count[a.unique_id] = (
                b.encounter_count.get(a.unique_id, 0) + boost
            )

            ## Institutional support makes the encounter less stressful.
            a.stress_level = max(0.0, a.stress_level - p["r5_stress_reduction"])
            b.stress_level = max(0.0, b.stress_level - p["r5_stress_reduction"])

            ## Structured contact creates mutual adaptation.
            a_old = a.cultural_vector.copy()
            b_old = b.cultural_vector.copy()
            a.cultural_vector += p["r5_learning_rate"] * (b_old - a_old)
            b.cultural_vector += p["r5_learning_rate"] * (a_old - b_old)
            a.cultural_vector = np.clip(a.cultural_vector, 0, 1)
            b.cultural_vector = np.clip(b.cultural_vector, 0, 1)

            ## Cross-country supported contact can also improve practical fluency.
            a.language_fluency = min(1.0, a.language_fluency + p["r5_language_gain"])
            b.language_fluency = min(1.0, b.language_fluency + p["r5_language_gain"])

            ## Once the structured encounter threshold is reached, strengthen the tie.
            if a.encounter_count[b.unique_id] >= p["r5_friendship_threshold"]:
                self.add_or_strengthen_social_tie(
                    a,
                    b,
                    amount=p["r5_tie_strength_increment"],
                )

    ## Define one model step
    def step(self):
        ## Reset pair tracker at the beginning of each tick
        self.interacted_pairs_this_tick = set()

        ## Existing ties fade first; later interactions can reinforce them again
        self.decay_social_ties()

        ## Agents act in random order, matching the class examples
        self.agents.shuffle_do("step")

        ## Apply the institutional pairing intervention after normal movement
        self.r5_pairing_intervention()

        ## Collect model data after all actions in the tick
        self.tick += 1
        self.datacollector.collect(self)


## Mean share of culturally similar neighbors around agents
def compute_spatial_clustering(model):
    ratios = []
    for agent in model.agents:
        neighbors = model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False
        )
        if not neighbors:
            continue

        similar = sum(
            1
            for n in neighbors
            if agent.cultural_distance(n) < model.params["r1_distance_threshold"]
        )
        ratios.append(similar / len(neighbors))

    return float(np.mean(ratios)) if ratios else 0.0


## Share of active social ties connecting students from the same country
def compute_network_homophily(model):
    graph = model.friendship_network
    if graph.number_of_edges() == 0:
        return 0.0

    id_to_country = {agent.unique_id: agent.country for agent in model.agents}
    same_country_edges = sum(
        1 for u, v in graph.edges() if id_to_country.get(u) == id_to_country.get(v)
    )

    return same_country_edges / graph.number_of_edges()


## Share of active social ties crossing country groups
def compute_cross_cultural_friendship_share(model):
    graph = model.friendship_network
    if graph.number_of_edges() == 0:
        return 0.0
    return 1.0 - compute_network_homophily(model)


## Average stress across all students
def compute_mean_stress(model):
    stresses = [agent.stress_level for agent in model.agents]
    return float(np.mean(stresses)) if stresses else 0.0


## Average stress only among students still active in campus interaction
def compute_mean_stress_active(model):
    stresses = [agent.stress_level for agent in model.agents if agent.state == "active"]
    return float(np.mean(stresses)) if stresses else 0.0


## Share of students temporarily isolated
def compute_isolation_rate(model):
    total = len(model.agents)
    if total == 0:
        return 0.0
    isolated = sum(1 for agent in model.agents if agent.state == "isolated")
    return isolated / total


## Average strength of active social ties
def compute_mean_tie_strength(model):
    graph = model.friendship_network
    if graph.number_of_edges() == 0:
        return 0.0

    weights = [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)]
    return float(np.mean(weights)) if weights else 0.0


## Exact mean pairwise cultural distance across all students
def compute_mean_cultural_distance(model):
    agents = list(model.agents)
    n = len(agents)
    if n < 2:
        return 0.0

    vectors = np.array([agent.cultural_vector for agent in agents])
    diff = vectors[:, None, :] - vectors[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2)) / np.sqrt(5)
    upper_i, upper_j = np.triu_indices(n, k=1)

    return float(np.mean(dist_matrix[upper_i, upper_j]))


## Run one model for a fixed number of ticks and return a dataframe
def run_single(
    scenario="baseline",
    intervention_enabled=True,
    seed=42,
    n_ticks=300,
    n_agents=500,
    width=50,
    height=50,
):
    model = CampusModel(
        n_agents=n_agents,
        width=width,
        height=height,
        population_scenario=scenario,
        intervention_enabled=intervention_enabled,
        seed=seed,
    )

    for _ in range(n_ticks):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().copy()
    df["scenario"] = scenario
    df["intervention"] = intervention_enabled
    df["seed"] = seed
    df["tick"] = df.index
    return df


## Use deterministic seeds rather than Python hash(), which can change by session
def deterministic_seed(base_seed, scenario_index, intervention_index, run_index):
    return int(base_seed + scenario_index * 10000 + intervention_index * 1000 + run_index)


## Run all scenario/intervention combinations for batch comparison
def run_batch_experiments(
    scenarios=("baseline", "chinese_heavy", "balanced"),
    interventions=(False, True),
    n_runs=20,
    n_ticks=300,
    n_agents=500,
    width=50,
    height=50,
    base_seed=1000,
    verbose=True,
):
    import pandas as pd

    results = []
    total = len(scenarios) * len(interventions) * n_runs
    completed = 0

    for scenario_index, scenario in enumerate(scenarios):
        for intervention_index, intervention in enumerate(interventions):
            for run_index in range(n_runs):
                seed = deterministic_seed(
                    base_seed, scenario_index, intervention_index, run_index
                )
                df = run_single(
                    scenario=scenario,
                    intervention_enabled=intervention,
                    seed=seed,
                    n_ticks=n_ticks,
                    n_agents=n_agents,
                    width=width,
                    height=height,
                )
                df["run"] = run_index
                results.append(df)

                completed += 1
                if verbose:
                    print(
                        f"[{completed}/{total}] scenario={scenario}, "
                        f"intervention={intervention}, run={run_index}, seed={seed} done"
                    )

    return pd.concat(results, ignore_index=True)


## Quick smoke test when model.py is run directly
if __name__ == "__main__":
    print("Running quick smoke test: baseline, intervention=True")
    test_results = run_single(n_ticks=100, seed=42)
    print(test_results.tail())