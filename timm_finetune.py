import timm
from src.utils import get_args_parser
import json
import pandas as pd
import csv
import os
import subprocess

if __name__ == "__main__":
    timm_models_csv = os.path.join('assets', 'timm_models.csv')
    timm_results_csv = os.path.join('assets', 'results-imagenet.csv')
    p_log_file = f'logs/perspective_results_ft.json'
    d_log_file = f'logs/depth_results_ft.json'
        
    df = pd.read_csv(timm_models_csv)
    timm_df = pd.read_csv(timm_results_csv)
    num_models = len(df.index)

    for index in df.index:
        with open(p_log_file, 'r') as f:
            perspective_results = json.load(f)
        with open(d_log_file, 'r') as f:
            depth_results = json.load(f)
            
        model_name = df['model_name'][index]
        if not model_name in timm_df['model'].values:
            continue
        
        if model_name not in perspective_results.keys() or perspective_results[model_name]=='Failed':
            print('persp', model_name)
            #subprocess.run(f'sbatch scripts/finetune.sh {model_name}', shell=True)
        
        if model_name not in depth_results.keys() or depth_results[model_name]=='Failed':
            #subprocess.run(f'sbatch scripts/finetune_depth.sh {model_name}', shell=True)
            print('depth', model_name)