import timm
import json
import pandas as pd
import csv
import os
import subprocess

if __name__ == "__main__":
    timm_models_csv = os.path.join('scripts', 'timm_models.csv')
    model_df = pd.read_csv(timm_models_csv)
    timm_results_csv = os.path.join('scripts', 'results-imagenet.csv')
    results_df = pd.read_csv(timm_results_csv)
    count = 0
    for index in model_df.index:
        model_name = model_df['model_name'][index]
        if not model_name in results_df['model'].values:
            continue
        img_size = results_df['img_size'][index]
        
        if not img_size == 224:
            print(model_name, img_size)
            count += 1
    print(count)