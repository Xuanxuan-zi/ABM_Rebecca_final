import numpy as np
from mesa import Agent


## Cultural centers are used only to initialize agents.
## They are not meant to say that everyone from one country is the same.
## Order: individualism, power distance, uncertainty avoidance,
## masculinity, long-term orientation. All values are scaled to 0-1.
HOFSTEDE_CENTERS = {
    "US": np.array([0.91, 0.40, 0.46, 0.62, 0.26]),
    "China": np.array([0.20, 0.80, 0.30, 0.66, 0.87]),
    "India": np.array([0.48, 0.77, 0.40, 0.56, 0.51]),
    "S.Korea": np.array([0.18, 0.60, 0.85, 0.39, 1.00]),
    "Germany": np.array([0.67, 0.35, 0.65, 0.66, 0.83]),
    "Brazil": np.array([0.38, 0.69, 0.76, 0.49, 0.44]),
}


## English fluency ranges by country.
## These are stylized assumptions for the model, not empirical estimates.
LANGUAGE_FLUENCY_RANGES = {
    "US": (0.95, 1.00),
    "China": (0.40, 0.80),
    "India": (0.70, 0.95),
    "S.Korea": (0.50, 0.80),
    "Germany": (0.75, 0.95),
    "Brazil": (0.50, 0.80),
}


class Student(Agent):
    ## Initiate student agent, inheriting the model property from Mesa Agent
    def __init__(self, model, country, cultural_sigma=0.08):
        super().__init__(model)

        ## Country is used to initialize culture and language, and later for summaries
        self.country = country

        ## Give each student a cultural vector near the country center
        center = HOFSTEDE_CENTERS[country]
        noise = self.model.rng.normal(0, cultural_sigma, size=5)
        self.cultural_vector = np.clip(center + noise, 0, 1)

        ## Give each student a random English fluency score within the country range
        fluency_low, fluency_high = LANGUAGE_FLUENCY_RANGES[country]
        self.language_fluency = float(
            self.model.rng.uniform(fluency_low, fluency_high)
        )

        ## Individual state variables
        self.stress_level = 0.0
        self.conflict_memory = 0.0
        self.state = "active"
        self.recovery_counter = 0

        ## Store repeated encounters with other agents by unique id
        self.encounter_count = {}

        ## Partners created through the university-supported R5 program.
        ## This is used only as a mild signal for mixed social spaces.
        self.structured_contact_partners = set()

    ## Calculate normalized Euclidean distance between two cultural vectors
    def cultural_distance(self, other):
        diff = self.cultural_vector - other.cultural_vector
        return float(np.linalg.norm(diff) / np.sqrt(5))

    ## Define effective language fluency for this pair of students
    def interaction_fluency(self, other):
        ## Same-country students are assumed to share enough language background
        ## for ordinary interaction, so English fluency only matters across countries.
        if self.country == other.country:
            return 1.0
        return min(self.language_fluency, other.language_fluency)

    ## Add one dyadic encounter for both students
    def record_encounter(self, other):
        other_id = other.unique_id
        self_id = self.unique_id
        self.encounter_count[other_id] = self.encounter_count.get(other_id, 0) + 1
        other.encounter_count[self_id] = other.encounter_count.get(self_id, 0) + 1
        return self.encounter_count[other_id]

    ## R1: culturally similar agents can build a maintained social tie
    def r1_social_tie(self, other, p):
        fluency = self.interaction_fluency(other)

        ## If the pair cannot communicate enough, no meaningful encounter happens
        if fluency < p["r1_absolute_min_fluency"]:
            return

        ## Similar interaction lowers stress for both agents
        stress_relief = p["r1_stress_reduction"] * fluency
        self.stress_level = max(0.0, self.stress_level - stress_relief)
        other.stress_level = max(0.0, other.stress_level - stress_relief)

        ## Count this as one shared encounter for the pair
        encounters = self.record_encounter(other)

        ## Lower effective fluency means more repeated encounters are needed
        required = int(
            np.ceil(p["r1_encounters_to_friend"] / max(fluency, 0.3))
        )

        ## Use >= so a boosted encounter count cannot accidentally skip the threshold
        if encounters >= required:
            self.model.add_or_strengthen_social_tie(
                self,
                other,
                amount=p["r1_tie_strength_increment"],
            )

    ## R2: moderate difference can lead to learning, with a small stress cost
    def r2_learning(self, other, p):
        fluency = self.interaction_fluency(other)

        ## If communication is not successful this time, no learning occurs
        if self.model.rng.random() > fluency:
            return

        ## Use copies so both updates use the old positions in cultural space
        self_old = self.cultural_vector.copy()
        other_old = other.cultural_vector.copy()

        self.cultural_vector += p["r2_learning_rate"] * (other_old - self_old)
        other.cultural_vector += p["r2_learning_rate"] * (self_old - other_old)

        self.cultural_vector = np.clip(self.cultural_vector, 0, 1)
        other.cultural_vector = np.clip(other.cultural_vector, 0, 1)

        ## English fluency only increases when the interaction crosses countries
        if self.country != other.country:
            self.language_fluency = min(
                1.0, self.language_fluency + p["r2_language_gain"]
            )
            other.language_fluency = min(
                1.0, other.language_fluency + p["r2_language_gain"]
            )

        ## Learning can be productive and still tiring
        self.stress_level = min(1.0, self.stress_level + p["r2_stress_cost"])
        other.stress_level = min(1.0, other.stress_level + p["r2_stress_cost"])

    ## Helper for R3: add conflict stress to one agent
    def add_conflict_stress(self, other, p):
        distance = self.cultural_distance(other)
        distance_factor = distance / max(p["r3_distance_threshold"], 0.01)
        memory_multiplier = 1.0 + self.conflict_memory
        language_penalty = 2.0 - self.interaction_fluency(other)

        stress_delta = (
            p["r3_base_stress"]
            * distance_factor
            * memory_multiplier
            * language_penalty
        )

        self.stress_level = min(1.0, self.stress_level + stress_delta)
        self.conflict_memory = min(
            1.0, self.conflict_memory + p["r3_memory_increment"]
        )

        ## High stress means the student temporarily withdraws from campus contact
        if self.stress_level >= p["r3_isolation_threshold"]:
            self.state = "isolated"

    ## R3: culturally distant interaction raises stress for both students
    def r3_conflict(self, other, p):
        self.add_conflict_stress(other, p)
        other.add_conflict_stress(self, p)

    ## Low English fluency can create background stress in cross-country settings
    def passive_language_stress(self, p):
        if self.language_fluency >= p["language_stress_threshold"]:
            return

        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        if not neighbors:
            return

        ## Same-country neighbors do not count as English-language stressors
        foreign_count = sum(
            1
            for neighbor in neighbors
            if neighbor.country != self.country
            and self.cultural_distance(neighbor) > p["r1_distance_threshold"]
        )
        foreign_ratio = foreign_count / len(neighbors)
        fluency_deficit = p["language_stress_threshold"] - self.language_fluency

        passive_stress = (
            p["passive_language_stress_rate"]
            * foreign_ratio
            * fluency_deficit
        )
        self.stress_level = min(1.0, self.stress_level + passive_stress)

    ## Stress and conflict memory fade somewhat over time
    def update_stress_lifecycle(self, p):
        self.stress_level *= p["stress_decay_rate"]
        self.conflict_memory *= p["conflict_memory_decay"]

        ## Isolated students can return after stress stays low for several ticks
        if self.state == "isolated":
            if self.stress_level < p["recovery_threshold"]:
                self.recovery_counter += 1
                if self.recovery_counter >= p["recovery_duration"]:
                    self.state = "active"
                    self.recovery_counter = 0
            else:
                self.recovery_counter = 0

    ## R4: move toward local areas with more culturally similar neighbors
    def move(self):
        p = self.model.params
        possibles = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        empty_cells = [cell for cell in possibles if self.model.grid.is_cell_empty(cell)]

        if not empty_cells:
            return

        weights = []
        for cell in empty_cells:
            similar_neighbors = self.count_similar_neighbors(
                cell, p["r1_distance_threshold"]
            )
            pairs = similar_neighbors // p["r4_cluster_pair_size"]
            weight = p["r4_weight_multiplier"] ** pairs

            ## High stress makes familiar spaces even more attractive
            if self.stress_level > p["r4_high_stress_threshold"]:
                weight *= p["r4_weight_multiplier"] ** pairs

            ## R5 can slightly weaken clustering by creating shared mixed spaces.
            ## This is intentionally mild, so the intervention mainly changes
            ## friendship networks rather than magically mixing the whole grid.
            structured_neighbors = self.count_structured_contact_neighbors(cell)
            if structured_neighbors > 0:
                weight *= p["r5_partner_movement_bonus"] ** structured_neighbors

            weights.append(weight)

        weights = np.array(weights, dtype=float)
        probabilities = weights / weights.sum()
        choice = int(self.model.rng.choice(len(empty_cells), p=probabilities))
        self.model.grid.move_agent(self, empty_cells[choice])

    ## Count culturally similar neighbors around a possible new location
    def count_similar_neighbors(self, cell, threshold):
        neighbors = self.model.grid.get_neighbors(
            cell, moore=True, include_center=False
        )
        return sum(
            1 for neighbor in neighbors if self.cultural_distance(neighbor) < threshold
        )

    ## Count R5 partners around a possible new location.
    ## The effect is deliberately small: structured programs create shared spaces,
    ## but they do not erase ordinary clustering preferences.
    def count_structured_contact_neighbors(self, cell):
        if not self.structured_contact_partners:
            return 0

        neighbors = self.model.grid.get_neighbors(
            cell, moore=True, include_center=False
        )
        return sum(
            1
            for neighbor in neighbors
            if neighbor.unique_id in self.structured_contact_partners
        )

    ## Apply the interaction rule to one neighbor
    def interact(self, other):
        p = self.model.params
        distance = self.cultural_distance(other)

        if distance < p["r1_distance_threshold"]:
            self.r1_social_tie(other, p)
        elif distance <= p["r3_distance_threshold"]:
            self.r2_learning(other, p)
        else:
            self.r3_conflict(other, p)

    ## One student step: move, interact with neighbors, then update stress
    def step(self):
        p = self.model.params

        ## Isolated agents do not move or interact in this tick
        if self.state == "isolated":
            self.update_stress_lifecycle(p)
            return

        self.move()

        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        for other in neighbors:
            if other.state == "isolated":
                continue

            ## Prevent A-B and B-A from being counted twice in the same tick
            pair_key = tuple(sorted((self.unique_id, other.unique_id)))
            if pair_key in self.model.interacted_pairs_this_tick:
                continue
            self.model.interacted_pairs_this_tick.add(pair_key)

            self.interact(other)

        self.passive_language_stress(p)
        self.update_stress_lifecycle(p)


## Default parameters for the model.
## These are baseline assumptions and can be varied in sensitivity checks.
DEFAULT_PARAMS = {
    ## R1: ordinary social tie formation
    "r1_distance_threshold": 0.30,
    "r1_absolute_min_fluency": 0.30,
    "r1_stress_reduction": 0.05,
    "r1_encounters_to_friend": 5,

    ## R2: cultural learning
    "r2_learning_rate": 0.002,
    "r2_language_gain": 0.005,
    "r2_stress_cost": 0.01,

    ## R3: conflict and isolation
    "r3_distance_threshold": 0.45,
    "r3_base_stress": 0.10,
    "r3_memory_increment": 0.05,
    "r3_isolation_threshold": 0.85,

    ## Passive language stress
    "language_stress_threshold": 0.70,
    "passive_language_stress_rate": 0.03,

    ## Stress lifecycle
    "stress_decay_rate": 0.98,
    "conflict_memory_decay": 0.995,
    "recovery_threshold": 0.40,
    "recovery_duration": 5,

    ## R4: snowball movement
    "r4_cluster_pair_size": 2,
    "r4_weight_multiplier": 1.3,
    "r4_high_stress_threshold": 0.5,

    ## R5: university-supported structured pairing intervention
    "r5_enabled": True,
    "r5_frequency": 20,
    "r5_coverage_rate": 0.20,
    "r5_pairs_per_event": 50,
    "r5_min_distance": 0.30,
    "r5_max_distance": 0.55,
    "r5_min_fluency": 0.40,
    "r5_encounter_boost": 4,
    "r5_stress_reduction": 0.05,
    "r5_learning_rate": 0.006,
    "r5_language_gain": 0.006,
    "r5_friendship_threshold": 4,
    "r5_partner_movement_bonus": 1.15,

    ## Maintained social tie lifecycle
    "r1_tie_strength_increment": 0.40,
    "r5_tie_strength_increment": 0.90,
    "tie_decay_rate": 0.993,
    "tie_removal_threshold": 0.20,
    "tie_max_strength": 4.00,
}
