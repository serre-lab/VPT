import csv
import os
import pandas as pd
import glob


if __name__ == "__main__":
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs/linear_probe_preds'
    vpt_results = glob.glob(os.path.join(result_directory, '*_preds_lp.csv'))
    depth_results = glob.glob(os.path.join(result_directory, '*_preds_depth_lp.csv'))
    combined_results = {}
    headers = ['img_name', 'vpt_label']
    
    for name in vpt_results:
        model_name = name.split('/')[-1]
        model_name = model_name.replace('_preds_human_lp.csv', '')
        headers.append(model_name)
        vpt_result = pd.read_csv(name).to_numpy()
        for line in vpt_result:
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
    with open(os.path.join(result_directory, 'combined_preds_human_lp.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)
    
    combined_results = {}
    headers = ['img_name', 'depth_label']        
    for name in depth_results:
        model_name = name.split('/')[-1]
        model_name = model_name.replace('_preds_depth_human_lp.csv', '')
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
    with open(os.path.join(result_directory, 'combined_preds_depth_human_lp.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)
    
        