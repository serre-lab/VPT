import os
import csv
import sys

def get_labels(data_dir):
    labels = [['img_url', 'label']]
    for category in os.listdir(data_dir):
        for scene in os.listdir(os.path.join(data_dir, category)):
            for i, setting in enumerate(sorted(os.listdir(os.path.join(data_dir, category, scene)))):
                print(i, setting)
                image_root = os.path.join(category, scene, setting)
                if i < 5:
                    label = 0.0
                else:
                    label = 1.0
                for img_name in os.listdir(os.path.join(data_dir, image_root)):
                    labels.append([os.path.join(image_root, img_name), label])
    return labels                    

if __name__ == "__main__":
    data_path = sys.argv[1]
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    
    train_labels = get_labels(train_dir)
    test_labels = get_labels(test_dir)
    
    train_csv_f = os.path.join(data_path, 'train_perspective.csv')
    test_csv_f = os.path.join(data_path, 'test_perspective.csv')
    
    with open(train_csv_f, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(train_labels)
    
    with open(test_csv_f, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(test_labels)
        