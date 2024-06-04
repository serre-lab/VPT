import os
import csv
import sys
import pandas as pd


if __name__ == "__main__":
    file_path = sys.argv[1]
    data_path = sys.argv[2]
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    img_labels = pd.read_csv(file_path).to_numpy()
    test_csv_content = [['img_url', 'label']]
    train_csv_content = [['img_url', 'label']]
    print(img_labels)
    for i in range(len(img_labels)):
        index = i
        scene_name = img_labels[index, 0]
        cat, scene, setting = scene_name.split('_')
        base_path = os.path.join(cat, cat+"_"+scene, setting)
        
        for j in range(len(img_labels[index])-1):
            
            img_id = str(j).zfill(4)
            img_index = j+1
            img_label = img_labels[index, img_index]
            if img_label == -1:
                continue
            img_path = os.path.join(base_path, img_id+".png")
            if scene == "03":
                if not os.path.isfile(os.path.join(test_path, img_path)):
                    print(img_path)
                    continue
                test_csv_content.append([img_path, float(img_label)])
            else:
                if not os.path.isfile(os.path.join(train_path, img_path)):
                    print(img_path)
                    continue
                train_csv_content.append([img_path, float(img_label)])
    with open(os.path.join(data_path, 'train_depth.csv'), 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(train_csv_content)
    
    with open(os.path.join(data_path, 'test_depth.csv'), 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(test_csv_content)
    