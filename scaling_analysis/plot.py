from custom_layers.custom_bert import BertForMaskedLM
import pandas as pd
from utils import count_parameters, calculate_params_from_config
import ast
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
import json
from transformers import BertConfig
from pprint import pprint

plotly.io.orca.config.save()


def convert_to_dict(string):
    _dict = json.loads(
        string.replace("BertConfig ", "").replace("\n", "").replace('""', '"')
    )
    return BertConfig(**_dict)


ikd = pd.read_csv("attention_best_ikd.csv")
biased = pd.read_csv("attention_best_biased.csv")
random = pd.read_csv("attention_best_random.csv")

ikd["config"] = ikd["config"].map(convert_to_dict)
biased["config"] = biased["config"].map(convert_to_dict)
random["config"] = random["config"].map(convert_to_dict)


params_lst = []
hover_templates = []
# model = BertForMaskedLM(config=ikd["config"][0])

sampling_dimensions = [
    "sample_hidden_size",
    "sample_num_attention_heads",
    "sample_intermediate_size",
    "sample_num_hidden_layers",
]

hidden_size_lst = []
inter_lst = []
hidden_layers_lst = []

for index, row in tqdm(ikd.iterrows()):
    config = row["config"]
    # model.set_sample_config(config)
    # _m = model.get_active_subnet(config)
    # params = count_parameters(_m)
    params = calculate_params_from_config(config)
    params_lst.append(params)

    hidden_size_lst.append(getattr(config, "sample_hidden_size"))
    hidden_layers_lst.append(getattr(config, "sample_num_hidden_layers"))

    hover_templates.append(
        "<br>".join([f"{key}: {getattr(config, key)}" for key in sampling_dimensions])
    )

config_dict = {
    "sample_hidden_size": 768,
    "sample_num_attention_heads": 12,
    "sample_intermediate_size": 3072,
    "sample_num_hidden_layers": 12,
}


hover_templates_global = [
    "<br>".join([f"{key}: {config_dict[key]}" for key in config_dict])
]


ikd["params"] = params_lst
biased["params"] = params_lst
random["params"] = params_lst


# ikd.to_csv("attention_best_ikd.csv", index=False)
# biased.to_csv("attention_best_biased.csv", index=False)
# random.to_csv("attention_best_random.csv", index=False)


fig = go.Figure()  # create a plotly figure

fig.add_trace(
    go.Scatter(
        x=ikd["params"].tolist(),
        y=ikd["perplexity"].tolist(),
        mode="markers",
        hovertext=hover_templates,
        marker=dict(size=4),  # color=hidden_layers_lst),
        # font_size=15,
        opacity=1,
        name="Biased Sampling+IKD",
    )
)
fig.add_trace(
    go.Scatter(
        x=biased["params"].tolist(),
        y=biased["perplexity"].tolist(),
        mode="markers",
        hovertext=hover_templates,
        marker=dict(size=4),  # color=hidden_layers_lst),
        # font_size=15,
        opacity=1,
        name="Biased Sampling",
    )
)
fig.add_trace(
    go.Scatter(
        x=random["params"].tolist(),
        y=random["perplexity"].tolist(),
        mode="markers",
        hovertext=hover_templates,
        marker=dict(size=4),  # color=hidden_layers_lst),
        opacity=1,
        # font_size=15,
        name="Random Sampling",
    )
)

### For the Super-transformer Perplexities ###
ikd_params = [130 * (10 ** 6)]

ikd_per = [3.625]
biased_per = [5.6]
random_per = [6.57]

fig.add_trace(
    go.Scatter(
        x=ikd_params,
        y=ikd_per,
        mode="markers",
        marker_symbol="star",
        hovertext=hover_templates_global,
        marker=dict(size=15),
        name="Biased Sampling+IKD Super",
    )
)
fig.add_trace(
    go.Scatter(
        x=ikd_params,
        y=biased_per,
        mode="markers",
        marker_symbol="star",
        hovertext=hover_templates_global,
        marker=dict(size=15),
        name="Biased Sampling Super",
    )
)
fig.add_trace(
    go.Scatter(
        x=ikd_params,
        y=random_per,
        mode="markers",
        marker_symbol="star",
        hovertext=hover_templates_global,
        marker=dict(size=15),
        name="Random Sampling Super",
    )
)

fig.update_layout(xaxis_title="Parameters", yaxis_title="Perplexity")

fig.show()
fig.write_image("spread_pareto.html")
