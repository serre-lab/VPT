import csv
import os
import pandas as pd
import glob
import json

if __name__ == "__main__":
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs/fine_tune_preds'
    vpt_results = glob.glob(os.path.join(result_directory, '*_preds_human_ft.csv'))
    depth_results = glob.glob(os.path.join(result_directory, '*_preds_depth_human_ft.csv'))
    combined_results = {}
    headers = ['img_name', 'vpt_label']
    for name in vpt_results:
        if "combined_preds" in name:
            continue
        model_name = name.split('/')[-1]
        model_name = model_name.replace('_preds_human_ft.csv', '')
        headers.append(model_name)
        vpt_result = pd.read_csv(name).to_numpy()
        for line in vpt_result:
            img_name = line[0]
            img = img_name.split('/')[-3:]
            img = '/'.join(img)
            if img_name not in combined_results.keys():
                print(img_name, name)
                combined_results[img_name] = [img, line[2], line[1]]
            else:
                combined_results[img_name].append(line[1])
    csv_output = [headers]
    
    for name in combined_results.keys():
        line = combined_results[name]
        csv_output.append(line)
    with open(os.path.join(result_directory, 'combined_preds_human_ft.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)
    
    combined_results = {}
    headers = ['img_name', 'depth_label']        
    for name in depth_results:
        model_name = name.split('/')[-1]
        model_name = model_name.replace('_preds_depth_human_ft.csv', '')
        headers.append(model_name)
        depth_result = pd.read_csv(name).to_numpy()
        for line in depth_result:
            img_name = line[0]
            img = img_name.split('/')[-3:]
            img = '/'.join(img)
            if img_name not in combined_results.keys():
                combined_results[img_name] = [img, line[2], line[1]]
            else:
                combined_results[img_name].append(line[1])
    csv_output = [headers]
    for name in combined_results.keys():
        line = combined_results[name]
        csv_output.append(line)
    with open(os.path.join(result_directory, 'combined_preds_depth_human_ft.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)
        
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs'
    depth_result_json = os.path.join(result_directory, 'depth_results_ft.json')
    perception_result_json = os.path.join(result_directory, 'perspective_results_ft.json')
    imagenet_result_csv = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/scripts/results-imagenet.csv'
    timm_models_csv = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/scripts/timm_models.csv'
    
    with open(depth_result_json, 'r') as f:
        depth_result = json.load(f)
    with open(perception_result_json, 'r') as f:
        perspective_result = json.load(f)
    
    imagenet_result = pd.read_csv(imagenet_result_csv)
    timm_models = pd.read_csv(timm_models_csv)
    
    combined_result = [['model_name', 'imagenet_accuracy', 'perspective_accuracy', 'depth_accuracy', 'perspective_accuracy_human', 'depth_accuracy_human']]
    
    for index in timm_models.index:
        model_name = timm_models['model_name'][index]
        if model_name not in imagenet_result['model'].values or model_name not in perspective_result.keys():
            continue
        if perspective_result[model_name] == 'Failed':
            perspective_result[model_name] = [None, None, None]
        if depth_result[model_name] == 'Failed':
            depth_result[model_name] = [None, None, None]
        imagenet_index = imagenet_result.index[imagenet_result['model'] == model_name].tolist()[0]
        result = [model_name, imagenet_result['top1'][imagenet_index], perspective_result[model_name][1], depth_result[model_name][1], perspective_result[model_name][3], depth_result[model_name][3]]
        combined_result.append(result)
    with open(os.path.join(result_directory, 'combined_results_ft.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(combined_result)
    
    
    
        