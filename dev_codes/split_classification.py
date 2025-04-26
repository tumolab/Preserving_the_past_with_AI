import os
import shutil
from sklearn.model_selection import train_test_split

input_path = 'path_to_source_folders'  # À modifier
dest_path = 'path_to_destination'     # À modifier

angles = ['0', '90', '180', '270']

for angle in angles:
    src_dir = os.path.join(input_path, angle)
    images = os.listdir(src_dir)

    train_val, test = train_test_split(images, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=2/9, random_state=42)

    split_data = {'train': train, 'val': val, 'test': test}

    for split, files in split_data.items():
        dst_dir = os.path.join(dest_path, split, angle)
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
