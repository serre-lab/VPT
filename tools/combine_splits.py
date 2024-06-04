import timm
import json
import pandas as pd
import csv
import os
import subprocess

if __name__ == "__main__":
    timm_models_csv = os.path.join('scripts', 'timm_models.csv')
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs'
    df = pd.read_csv(timm_models_csv)
    depth_result_json_1 = os.path.join(result_directory, 'depth_results_first.json')
    perspective_result_json_1 = os.path.join(result_directory, 'perspective_results_first.json')
    depth_result_json_2 = os.path.join(result_directory, 'depth_results_second.json')
    perspective_result_json_2 = os.path.join(result_directory, 'perspective_results_second.json')
    with open(depth_result_json_1, 'r') as f:
        depth_result_1 = json.load(f)
    with open(depth_result_json_2, 'r') as f:
        depth_result_2 = json.load(f)
    with open(perspective_result_json_1, 'r') as f:
        perspective_result_1 = json.load(f)
    with open(perspective_result_json_2, 'r') as f:
        perspective_result_2 = json.load(f)    
    depth_result = {}
    perspective_result = {}
    for index in df.index:
        model_name = df['model_name'][index]
        if model_name in depth_result_1.keys():
            depth_result[model_name] = depth_result_1[model_name]
        if model_name in depth_result_2.keys():
            depth_result[model_name] = depth_result_2[model_name]
        if model_name in perspective_result_1.keys():
            perspective_result[model_name] = perspective_result_1[model_name]
        if model_name in perspective_result_2.keys():
            perspective_result[model_name] = perspective_result_2[model_name]
    
    with open(os.path.join(result_directory, 'depth_results.json'), 'w') as f:
        results_json = json.dumps(depth_result, indent=4)
        f.write(results_json)
    with open(os.path.join(result_directory, 'perspective_results.json'), 'w') as f:
        results_json = json.dumps(perspective_result, indent=4)
        f.write(results_json)