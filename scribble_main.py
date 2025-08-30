import random
import os
from data_loader import load_dataset_from_txt
from active_learning_scribble import active_learning_loop_scribble  

# Load spilted data list
split_dir = "splits_1000"

# Get image address
train_gen, val_gen, test_gen, test_input_paths, test_target_paths = load_dataset_from_txt(
    split_dir=split_dir,
    img_size=(160,160),
    batch_size=4
)

# Get training images and ground truths address
all_img_paths = train_gen.input_img_paths + val_gen.input_img_paths
all_mask_paths = train_gen.target_img_paths + val_gen.target_img_paths

# Random spilt initial labeled datas and unlabeled datas
combined = list(zip(all_img_paths, all_mask_paths))
random.seed(1337)
random.shuffle(combined)

initial_labeled_images = [x[0] for x in combined[:200]]
initial_labeled_masks  = [x[1] for x in combined[:200]]
unlabeled_images       = [x[0] for x in combined[200:]]

# Active learning loop
active_learning_loop_scribble(
    initial_labeled_images=initial_labeled_images,
    initial_labeled_masks=initial_labeled_masks,
    unlabeled_images=unlabeled_images,
    scribble_dir="D:/project/CODE/scribbles/",  
    test_gen=test_gen,
    test_input_paths=test_input_paths,
    test_target_paths=test_target_paths,
    rounds=10,
    query_size=20,
    save_dir="D:/project/CODE/scribble_models/",
    initial_model_path="D:/project/CODE/oxford_segmentation.h5"
)
