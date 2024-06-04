import json
import csv
import os
import pandas as pd 

if __name__ == "__main__":
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs'
    depth_result_json = os.path.join(result_directory, 'depth_results.json')
    perception_result_json = os.path.join(result_directory, 'perspective_results.json')
    imagenet_result_csv = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/scripts/results-imagenet.csv'
    timm_models_csv = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/scripts/timm_models.csv'
    
    with open(depth_result_json, 'r') as f:
        depth_result = json.load(f)
    with open(perception_result_json, 'r') as f:
        perception_result = json.load(f)
    
    imagenet_result = pd.read_csv(imagenet_result_csv)
    timm_models = pd.read_csv(timm_models_csv)
    
    combined_result = [['model_name', 'imagenet_accuracy', 'perspective_accuracy', 'depth_accuracy', 'perspective_accuracy_human', 'depth_accuracy_human']]
    
    for index in timm_models.index:
        model_name = timm_models['model_name'][index]
        if model_name not in imagenet_result['model'].values:
            continue
        if perception_result[model_name] == 'Failed':
            perception_result[model_name] = [None, None, None]
        if depth_result[model_name] == 'Failed':
            depth_result[model_name] = [None, None, None]
        imagenet_index = imagenet_result.index[imagenet_result['model'] == model_name].tolist()[0]
        result = [model_name, imagenet_result['top1'][imagenet_index], perception_result[model_name][1], depth_result[model_name][1], perception_result[model_name][3], depth_result[model_name][3]]
        combined_result.append(result)
    with open(os.path.join(result_directory, 'combined_results.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(combined_result)
    