from custom_layers.custom_bert import BertForMaskedLM
import pandas as pd
from utils import count_parameters
import ast
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
import json
from transformers import AutoConfig
from pprint import pprint

plotly.io.orca.config.save()

def convert_to_dict(string):
    _dict = json.loads(string.replace("BertConfig ", "").replace('\n', '').replace('""', '"'))
    config = AutoConfig("bert-base-uncased")
    for key, value in _dict.items():
        setattr(config, key, value)
    return value

ikd = pd.read_csv("scaling_analysis/attention_best_ikd.csv")
biased = pd.read_csv("scaling_analysis/attention_best_biased.csv")
random = pd.read_csv("scaling_analysis/attention_best_random.csv")

ikd['config'] = ikd['config'].map(convert_to_dict)
biased['config'] = biased['config'].map(convert_to_dict)
random['config'] = random['config'].map(convert_to_dict)


<<<<<<< HEAD
ikd = ikd.head()
biased = biased.head()
random = random.head()
params_lst = []
=======
ikd = ikd.head(10)
biased = biased.head(10)
random = random.head(10)
params_lst = [] 
>>>>>>> 8a4390d235f47fcf925dcc5763414e1d6ece913c
model = BertForMaskedLM(config=ikd['config'][0])

for index, row in tqdm(ikd.iterrows()):
    config = row['config']
    model.set_sample_config(config)
    _m = model.get_active_subnet(config)
    params = count_parameters(_m)
    params_lst.append(params)


ikd["params"] = params_lst
biased["params"] = params_lst
random["params"] = params_lst


#ikd.to_csv("attention_best_ikd.csv", index=False)
#biased.to_csv("attention_best_biased.csv", index=False)
#random.to_csv("attention_best_random.csv", index=False)


fig = go.Figure() #create a plotly figure

fig.add_trace(go.Scatter(x=ikd["params"].tolist(), y=ikd["perplexity"].tolist(), mode='markers', font_size=15, name="Biased Sampling+IKD"))
fig.add_trace(go.Scatter(x=biased["params"].tolist(), y=biased["perplexity"].tolist(), mode='markers', font_size=15, name="Biased Sampling"))
fig.add_trace(go.Scatter(x=random["params"].tolist(), y=random["perplexity"].tolist(), mode='markers', font_size=15, name="Random Sampling"))

<<<<<<< HEAD
### For the Super-transformer Perplexities ###
ikd_params = [110*(10**6)]
=======
### For the Super-transformer Perplexities ### 
ikd_params = [108*(10**6)]
>>>>>>> 8a4390d235f47fcf925dcc5763414e1d6ece913c

ikd_per = [3.625]
biased_per = [5.6]
random_per = [6.57]

fig.add_trace(go.Scatter(x=ikd_params, y=ikd_per, mode='markers', marker_symbol='star', font_size=18, name="Biased Sampling+IKD Super"))
fig.add_trace(go.Scatter(x=ikd_params, y=biased_per, mode='markers', marker_symbol='star', font_size=18, name="Biased Sampling Super"))
fig.add_trace(go.Scatter(x=ikd_params, y=random_per, mode='markers', marker_symbol='star', font_size=18, name="Random Sampling Super"))

fig.update_layout(xaxis_title="Parameters", yaxis_title="Perplexity")

fig.show()
fig.write_image("spread_pareto.pdf")
