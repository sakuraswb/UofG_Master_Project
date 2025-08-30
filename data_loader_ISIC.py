import os
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow import keras

class ISICDataset(keras.utils.Sequence):
    
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = list(input_img_paths)
        self.target_img_paths = list(target_img_paths)

    def __len__(self):        
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i:i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i:i + self.batch_size]

        # input images
        x = np.zeros((len(batch_input_img_paths),) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)  
            img = np.array(img, dtype="float32") / 255.0
            x[j] = img

        # ground truth
        y = np.zeros((len(batch_target_img_paths),) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            if isinstance(path, np.ndarray):
                mask = path.astype("float32")
            else:
                mask_img = Image.open(path).convert("L")  
                mask_img = mask_img.resize(self.img_size, resample=Image.NEAREST)  
                mask = np.array(mask_img, dtype="float32")
                mask = (mask > 127).astype("float32")
            y[j] = np.expand_dims(mask, axis=-1)

        return x, y


def split_dataset(input_dir, target_dir, img_size=(160,160), batch_size=8, max_images=2000, save_split_dir="splits"):        
    input_img_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith((".jpg", ".png")) and not fname.startswith(".")
    ])
    target_img_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.lower().endswith((".gif", ".png")) and not fname.startswith(".")
    ])

    # Image alignment check
    if len(input_img_paths) != len(target_img_paths):
        print(f" images number {len(input_img_paths)} and ground truth number {len(target_img_paths)} are inconsistent")

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

    
    train_data = ISICDataset(batch_size, img_size, input_img_paths[:n_train], target_img_paths[:n_train])
    val_data = ISICDataset(batch_size, img_size, input_img_paths[n_train:n_train+n_val], target_img_paths[n_train:n_train+n_val])
    test_data = ISICDataset(batch_size, img_size, input_img_paths[n_train+n_val:], target_img_paths[n_train+n_val:])

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

    train_data = ISICDataset(batch_size, img_size, train_images, train_masks)
    val_data   = ISICDataset(batch_size, img_size, val_images, val_masks)
    test_data  = ISICDataset(batch_size, img_size, test_images, test_masks)

    return train_data, val_data, test_data, test_images, test_masks
