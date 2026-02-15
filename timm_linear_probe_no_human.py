import timm
from src.utils import get_args_parser
import json
from run_linear_probe_no_human import run_linear_probe
import pandas as pd
import csv
import os
import subprocess
import time
import torch
import shutil

huggingface_path = "/users/bkim53/.cache/huggingface/hub"

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    timm_models_csv = os.path.join('assets', 'timm_models.csv')
    timm_results_csv = os.path.join('assets', 'results-imagenet.csv')
    if args.task == 'perspective':
        log_file = f'logs/perspective_results_nh.json'
    else:
        # log_file = f'logs/depth_results_{args.split}.json'
        log_file = f'logs/{args.task}_results_nh.json'
        
    df = pd.read_csv(timm_models_csv)
    timm_df = pd.read_csv(timm_results_csv)
    num_models = len(timm_df.index)
    if args.split == 'first':
        models_range = range(int(len(df.index)/2))
    elif args.split == 'second':
        models_range = range(int(len(df.index)/2), len(df.index))
    else:
        models_range = range(len(df.index))
    for index in models_range:
        torch.cuda.empty_cache()
        with open(log_file, 'r') as f:
            perspective_results = json.load(f)
        #print(df['model'][index], df['top1'][index], df['img_size'][index])
        model_name = df['model_name'][index]
        # model_name = timm_df['model'][index]
        #img_size = df['img_size'][index]
        if not model_name in timm_df['model'].values:
            continue
        print(model_name)
        if model_name in perspective_results.keys() and perspective_results[model_name]!='Failed':
            continue
        print(model_name)
        args.model_name = model_name
        start = time.time()
        try:
            run_linear_probe(args)
        except:
            perspective_results[model_name] = "Failed"
            with open(log_file, 'w') as f:
                results_json = json.dumps(perspective_results, indent=4)
                f.write(results_json)
        end = time.time()
        print(f"TIME ELAPSED: {end - start} seconds")
        print("Clearing out huggingface hub cache...")
        for name in os.listdir(huggingface_path):
            subpath = os.path.join(huggingface_path, name)
            if os.path.isdir(subpath) and os.path.isdir(os.path.join(huggingface_path, name)):
                shutil.rmtree(subpath)
        print("Cleared huggingface hub cache")