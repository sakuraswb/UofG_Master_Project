import os
import random
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow import keras

class OxfordPets(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i:i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i:i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            if isinstance(path, np.ndarray):                
                mask = path.astype("uint8")
            else:                
                mask = load_img(path, target_size=self.img_size, color_mode="grayscale")
                mask = np.array(mask).astype("uint8") - 1  
        
            y[j] = np.expand_dims(mask, axis=-1)
        return x, y

def split_dataset(input_dir, target_dir, img_size=(160,160), batch_size=8, max_images=None, save_split_dir="splits"):
    input_img_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir) if fname.endswith(".jpg")
    ])
    target_img_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ])

    # shuffle the order
    combined = list(zip(input_img_paths, target_img_paths))
    random.seed(1337)
    if max_images is not None and max_images < len(combined):
        combined = random.sample(combined, max_images)
    random.shuffle(combined)

    input_img_paths, target_img_paths = zip(*combined)

    total = len(input_img_paths)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)

    os.makedirs(save_split_dir, exist_ok=True)

    def save_list(file_path, data_list):
        with open(file_path, "w") as f:
            for path in data_list:
                f.write(path + "\n")


    save_list(os.path.join(save_split_dir, "train.txt"), input_img_paths[:n_train])
    save_list(os.path.join(save_split_dir, "val.txt"), input_img_paths[n_train:n_train+n_val])
    save_list(os.path.join(save_split_dir, "test.txt"), input_img_paths[n_train+n_val:])

    save_list(os.path.join(save_split_dir, "train_mask.txt"), target_img_paths[:n_train])
    save_list(os.path.join(save_split_dir, "val_mask.txt"), target_img_paths[n_train:n_train+n_val])
    save_list(os.path.join(save_split_dir, "test_mask.txt"), target_img_paths[n_train+n_val:])

    train_data = OxfordPets(batch_size, img_size, input_img_paths[:n_train], target_img_paths[:n_train])
    val_data = OxfordPets(batch_size, img_size, input_img_paths[n_train:n_train+n_val], target_img_paths[n_train:n_train+n_val])
    test_data = OxfordPets(batch_size, img_size, input_img_paths[n_train+n_val:], target_img_paths[n_train+n_val:])

    return train_data, val_data, test_data, input_img_paths[n_train+n_val:], target_img_paths[n_train+n_val:]

def load_paths_from_txt(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_dataset_from_txt(split_dir, img_size=(160,160), batch_size=8):
    
    train_images = load_paths_from_txt(os.path.join(split_dir, "train.txt"))
    val_images   = load_paths_from_txt(os.path.join(split_dir, "val.txt"))
    test_images  = load_paths_from_txt(os.path.join(split_dir, "test.txt"))

    train_masks  = load_paths_from_txt(os.path.join(split_dir, "train_mask.txt"))
    val_masks    = load_paths_from_txt(os.path.join(split_dir, "val_mask.txt"))
    test_masks   = load_paths_from_txt(os.path.join(split_dir, "test_mask.txt"))

    train_data = OxfordPets(batch_size, img_size, train_images, train_masks)
    val_data   = OxfordPets(batch_size, img_size, val_images, val_masks)
    test_data  = OxfordPets(batch_size, img_size, test_images, test_masks)

    return train_data, val_data, test_data, test_images, test_masks