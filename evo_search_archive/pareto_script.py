import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tasks = ['sst2', 'mrpc', 'qqp', 'qnli', 'rte']
devices = ['xeon', 'gtx1080']
col = {'xeon':'blue', 'gtx1080':'red'}
dirname = 'evo_search_results_'
for task in tasks:
    for device in devices:
        plt.title(f'Pareto frontiers for GLUE task: {task}')
        plt.xlabel('latency')
        plt.ylabel('accuracy')
        x = []
        y = []

        # Get all points from the evosearch dump:
        for i in range(5):
            filename = f'{dirname}{device}/{task}/evo_dump_lat_{i}.txt'
            file = open(filename, 'r')
            lines = file.readlines()
            accuracy = float(lines[1].strip())
            latency = float(lines[2].strip())
            x.append(latency)
            y.append(accuracy)

        # removing interior points of the pareto:
        ind = np.argsort(np.array(x))
        x = list(np.array(x)[ind])
        y = list(np.array(y)[ind])
        xn = [x[0]]
        yn = [y[0]]
        for i in range(1, 5):
            if yn[-1] < y[i]:
                xn.append(x[i])
                yn.append(y[i])
        
        # Plot all the points according to device:
        plt.plot(
            xn,
            yn,
            color=col[device], 
            linestyle='dashed', 
            linewidth = 3, 
            marker='o', 
            markerfacecolor='black', 
            markersize=10, 
            label=f'pareto of {device}'
        )
        plt.legend()
        plt.savefig(f'Paretos/{task}_{device}')
        plt.close()


