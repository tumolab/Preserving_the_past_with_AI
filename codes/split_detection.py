from sklearn.model_selection import train_test_split
import os
import shutil

def split_dataset_sklearn(images_dir, labels_dir, output_dir):
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    train_files, temp_files = train_test_split(images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=1/3, random_state=42)

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split in splits:
        for subfolder in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subfolder), exist_ok=True)

        for img_file in splits[split]:
            label_file = os.path.splitext(img_file)[0] + '.txt'

            shutil.copy(os.path.join(images_dir, img_file), os.path.join(output_dir, split, 'images', img_file))
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dir, split, 'labels', label_file))

    return splits

split_dataset_sklearn("path/to/images", "path/to/labels", "path/to/output")
