import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

tasks = ['sst2', 'mrpc', 'qqp', 'qnli', 'rte']
devices = ['xeon', 'gtx1080']
dirname = 'evo_search_results_'

for task in tasks:
    for device in devices:
        # Get all points from the evosearch dump:
        tables_1 = []
        for i in range(5):
            filename = f'{dirname}{device}/{task}/evo_dump_lat_{i}.txt'
            file = open(filename, 'r')
            lines = file.readlines()
            config = eval(lines[0].strip())
            accuracy = float(lines[1].strip())
            latency = float(lines[2].strip())
            params = int(lines[3].strip())

            tables_1.append([i, round(latency, 2), round(accuracy*100, 2), params])
        
        tables_1 = np.array(tables_1)
        cols = [
            'Constraint level',
            'Latency',
            'Accuracy',
            'Best config total parameters',
        ]
        pd.DataFrame(data=tables_1, columns=cols).to_csv(f'tables/tables_1/{task}_{device}.csv', index=False)


tasks = ['sst2', 'mrpc', 'qqp', 'qnli', 'rte']
devices = ['xeon', 'gtx1080']
dirname = 'evo_search_results_'

for task in tasks:
    for i in range(5):
        tables_2 = []
        for device in devices:
        # Get all points from the evosearch dump:
            filename = f'{dirname}{device}/{task}/evo_dump_lat_{i}.txt'
            file = open(filename, 'r')
            lines = file.readlines()
            config = eval(lines[0].strip())['encoder']
            accuracy = float(lines[1].strip())
            latency = float(lines[2].strip())
            params = int(lines[3].strip())

            tables_2.append(
                [
                    device, 
                    params, 
                    config['encoder_layer_num'], 
                    config['encoder_embed_dim'],
                    round(np.mean(config['encoder_ffn_embed_dim']), 2),
                    round(np.mean(config['encoder_self_attention_heads']), 2),
                ]
            )
            
        tables_2 = np.array(tables_2)
        cols = [
            'Device',
            'Best config total parameters',
            'Num layers',
            'Embedding size',
            'Average FFN hidden size',
            'Average number of heads',
        ]
        pd.DataFrame(data=tables_2, columns=cols).to_csv(f'tables/tables_2/{task}_{i}.csv', index=False)
