import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import seaborn as sns
sns.set_style('whitegrid')


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
        

def load_dict_to_json(d, json_path):
    """Loads dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path) as f:
        params = json.load(f)
        d.update(params)
        return d


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics.json`
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    metrics_file = os.path.join(parent_dir, 'metrics.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)    
            

def synthesize_results(parent_dir):
    """Aggregates results from the metrics.json in a parent folder
    Args:
        parent_dir: (string) path to directory containing experiments results
    """
    metrics = dict()
    aggregate_metrics(parent_dir, metrics)
    metric_names = metrics[list(metrics.keys())[0]].keys()

    experiments = dict()
    for subdir, values in metrics.items():
        split_path = subdir.replace(parent_dir, '').split('/')
        param_name = split_path[1]
        split_exp = split_path[2].split('_')
        alg, param_val, run = split_exp[0], split_exp[1], split_exp[2]

        if param_name not in experiments:
            experiments[param_name] = dict()
        for metric_name in metric_names:
            if metric_name not in experiments[param_name]:
                experiments[param_name][metric_name] = dict()
            if alg not in experiments[param_name][metric_name]:
                experiments[param_name][metric_name][alg] = dict()
            if param_val not in experiments[param_name][metric_name][alg]:
                experiments[param_name][metric_name][alg][param_val] = dict()
            if not experiments[param_name][metric_name][alg][param_val]:
                experiments[param_name][metric_name][alg][param_val] = []
            experiments[param_name][metric_name][alg][param_val].append(values[metric_name])

    for param_name in experiments:
        cur_param = experiments[param_name]
        for metric_name in cur_param:
            cur_metric = cur_param[metric_name]
            fig = plt.figure()
            for alg in cur_metric:
                cur_lag = cur_metric[alg]
                param_val_list = []
                metric_val_list = []
                metric_std_list = []
                for param_val in cur_lag:
                    param_val_list.append(int(param_val))
                    metric_val_list.append(np.mean(cur_metric[alg][param_val]))
                    metric_std_list.append(np.std(cur_metric[alg][param_val]))
                param_val_list, metric_val_list = zip(*sorted(zip(param_val_list, metric_val_list)))

                param_val_np = np.asarray(param_val_list)
                metric_val_np = np.asarray(metric_val_list)
                metric_std_np = np.asarray(metric_std_list)
                if alg=='semi-supervised':
                    alg = 'FixMatch'
                elif alg=='consistency':
                    alg = 'Consistency Loss'
                elif alg=='supervised':
                    alg = 'Supervised'
                elif alg=='confidence-map':
                    alg = 'Confidence Map'
                plt.fill_between(param_val_np, metric_val_np-metric_std_np, metric_val_np+metric_std_np, alpha=0.2)
                plt.plot(param_val_np, metric_val_np, label=alg)

            plt.xlabel(param_name.replace('_', ' ').title())
            plt.ylabel(metric_name.replace('_', ' ').title())
            plt.legend()
            plt.show()
            fig.savefig(os.path.join(parent_dir, param_name, metric_name.replace(' ', '_') + '.png'))
            
    metrics = dict()
    aggregate_metrics(parent_dir, metrics)
    headers = metrics[list(metrics.keys())[0]].keys()
    table_list = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    table = tabulate(table_list, headers, tablefmt='pipe')
    
    save_file = os.path.join(parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)