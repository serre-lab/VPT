import json
import csv
import os
import pandas as pd 

def save_csv(result_directory):
    result_json = os.path.join(result_directory, 'view_results.json')
    imagenet_result_csv = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/scripts/results-imagenet.csv'
    timm_models_csv = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/scripts/timm_models.csv'
    with open(result_json, 'r') as f:
        view_result = json.load(f)
    
    imagenet_result = pd.read_csv(imagenet_result_csv)
    timm_models = pd.read_csv(timm_models_csv)
    
    combined_result = [['model_name', 'imagenet_accuracy', 'view_accuracy']]
    
    for index in timm_models.index:
        model_name = timm_models['model_name'][index]
        if model_name not in imagenet_result['model'].values or model_name not in view_result.keys():
            continue
        imagenet_index = imagenet_result.index[imagenet_result['model'] == model_name].tolist()[0]
        if isinstance(view_result[model_name], list):
            view_accuracy = view_result[model_name][-1]
        else:
            view_accuracy = view_result[model_name]
        result = [model_name, imagenet_result['top1'][imagenet_index], view_accuracy]
        combined_result.append(result)
    with open(os.path.join(result_directory, 'view_results.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(combined_result)
        
if __name__ == "__main__":
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs/linear_probe_view_exp'
    save_csv(result_directory)
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs/finetune_view_exp'
    save_csv(result_directory)
    