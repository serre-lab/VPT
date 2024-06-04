import timm
from src.utils import get_args_parser
import json
from test_finetune import run_evaluate
import pandas as pd
import csv
import os
import subprocess

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    timm_models_csv = os.path.join('assets', 'timm_models.csv')
    timm_results_csv = os.path.join('assets', 'results-imagenet.csv')
    log_file = f'./logs/finetune_view_exp/view_results.json'
    ckpt_root = './logs/fine_tune_ckpts'
    df = pd.read_csv(timm_models_csv)
    timm_df = pd.read_csv(timm_results_csv)
    num_models = len(df.index)
    
    models_range = range(len(df.index))
    for index in models_range:
        with open(log_file, 'r') as f:
            perspective_results = json.load(f)
        model_name = df['model_name'][index]
        if not model_name in timm_df['model'].values:
            continue
        if model_name in perspective_results.keys() and perspective_results[model_name]!='Failed':
            continue
        args.model_name = model_name
        args.ckpt_path = os.path.join(ckpt_root, f'{model_name}_perspective.ckpt')
        print(model_name)
        try:
            run_evaluate(args)
        except:
            perspective_results[model_name] = "Failed"
            with open(log_file, 'w') as f:
                results_json = json.dumps(perspective_results, indent=4)
                f.write(results_json)