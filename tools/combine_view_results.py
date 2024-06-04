import csv
import os
import pandas as pd
import glob
import json

if __name__ == "__main__":
    result_directory = '/oscar/data/tserre/Users/pzhou10/gs-perception/gs-perception/logs/'
    lp_results = glob.glob(os.path.join(result_directory, 'linear_probe_view_exp/*_view_lp.csv'))
    ft_results = glob.glob(os.path.join(result_directory, 'finetune_view_exp/*_view_ft.csv'))
    
    combined_ft = {}
    combined_lp = {}
    headers = ['img_name', 'label']
    
    for name in lp_results:
        if "combined_view" in name:
            continue
        model_name = name.split('/')[-1]
        model_name = model_name.replace('_preds_view_lp.csv', '')
        headers.append(model_name)
        lp_result = pd.read_csv(name).to_numpy()
        for line in lp_result:
            img_name = line[0]
            img = img_name.split('/')[-1]
            print(img)
            if img_name not in combined_lp.keys():
                combined_lp[img_name] = [img, line[2], line[1]]
            else:
                combined_lp[img_name].append(line[1])
                
    csv_output = [headers]
    for name in combined_lp.keys():
        line = combined_lp[name]
        csv_output.append(line)
    with open(os.path.join(result_directory, 'combined_view_lp.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)
        
    headers = ['img_name', 'label']    
    for name in ft_results:
        model_name = name.split('/')[-1]
        model_name = model_name.replace('_view_ft.csv', '')
        headers.append(model_name)
        ft_result = pd.read_csv(name).to_numpy()
        for line in ft_result:
            img_name = line[0]
            img = img_name.split('/')[-1]
            print(img)
            if img_name not in combined_ft.keys():
                combined_ft[img_name] = [img, line[2], line[1]]
            else:
                combined_ft[img_name].append(line[1])
                
    csv_output = [headers]
    for name in combined_ft.keys():
        line = combined_ft[name]
        csv_output.append(line)
    with open(os.path.join(result_directory, 'combined_view_ft.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_output)
        
    
    
    
        