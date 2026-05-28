from model import CampusModel
from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)
from mesa.visualization.user_param import Slider
from mesa.visualization.components import AgentPortrayalStyle


## Colors for each country group in the visualization
COUNTRY_COLORS = {
    "US": "blue",
    "China": "red",
    "India": "orange",
    "S.Korea": "green",
    "Germany": "purple",
    "Brazil": "brown",
}


## Define agent portrayal: color by country, smaller marker if isolated
def agent_portrayal(agent):
    if agent.state == "isolated":
        size = 8
        alpha = 0.35
    else:
        size = 14
        alpha = max(0.35, 1.0 - 0.6 * agent.stress_level)

    return AgentPortrayalStyle(
        color=COUNTRY_COLORS.get(agent.country, "gray"),
        marker="o",
        size=size,
        alpha=alpha,
    )


## Enumerate model parameters shown in SolaraViz
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "n_agents": Slider("Number of Agents", value=500, min=100, max=800, step=50),
    "width": 50,
    "height": 50,

    ## Important:
    ## Do not name this parameter "scenario".
    ## "scenario" may conflict with Mesa/Solara's internal Scenario object during reset.
    "population_scenario": {
        "type": "Select",
        "value": "baseline",
        "values": ["baseline", "chinese_heavy", "balanced"],
        "label": "Population Scenario",
    },

    "intervention_enabled": {
        "type": "Checkbox",
        "value": True,
        "label": "R5 Structured Pairing Intervention",
    },
}


## Instantiate model with the same parameter names as CampusModel
model = CampusModel(
    n_agents=500,
    width=50,
    height=50,
    population_scenario="baseline",
    intervention_enabled=True,
    seed=42,
)


## Define space and plot components
CampusSpace = make_space_component(agent_portrayal, draw_grid=False)

SegregationPlot = make_plot_component(
    {"spatial_clustering": "tab:blue"}
)

StressPlot = make_plot_component(
    {
        "mean_stress": "tab:orange",
        "isolation_rate": "tab:purple",
    }
)

FriendshipCompositionPlot = make_plot_component(
    {
        "network_homophily": "tab:red",
        "cross_cultural_friendship_share": "tab:cyan",
    }
)

SocialTiePlot = make_plot_component(
    {"active_social_ties": "tab:green"}
)

TieStrengthPlot = make_plot_component(
    {"mean_tie_strength": "tab:gray"}
)

CulturalDistancePlot = make_plot_component(
    {"mean_cultural_dist": "tab:brown"}
)


## Instantiate page including all components
page = SolaraViz(
    model,
    components=[
        CampusSpace,
        SegregationPlot,
        StressPlot,
        FriendshipCompositionPlot,
        SocialTiePlot,
        TieStrengthPlot,
        CulturalDistancePlot,
    ],
    model_params=model_params,
    name="International Students Cultural Segregation Model",
)


## Required by Solara
page