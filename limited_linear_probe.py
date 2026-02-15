import timm
from src.utils import get_args_parser
import json
from linear_probe_results import run_linear_probe
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
    
    models = [
        "resnext26ts.ra2_in1k",
        "resnext26ts.ra2_in1k",
        "resnext50_32x4d.gluon_in1k",
        "hrnet_w18_small.ms_in1k",
        "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
    ]
    
    for model_name in models:
        print(model_name)
        args.model_name = model_name
        start = time.time()
        try:
            run_linear_probe(args)
        except Exception as e:
            print(e)
        end = time.time()
        print(f"TIME ELAPSED: {end - start} seconds")
        print("Clearing out huggingface hub cache...")
        for name in os.listdir(huggingface_path):
            subpath = os.path.join(huggingface_path, name)
            if os.path.isdir(subpath) and os.path.isdir(os.path.join(huggingface_path, name)):
                shutil.rmtree(subpath)
        print("Cleared huggingface hub cache")