import random
from data_loader_ISIC import load_dataset_from_txt
from active_learning_ISIC import active_learning_loop_ISIC

# Load spilted data list
split_dir = "splits_ISIC_2000"

# 1. Load spilted data
train_gen, val_gen, test_gen, test_input_paths, test_target_paths = load_dataset_from_txt(
    split_dir=split_dir,
    img_size=(160, 160),
    batch_size=4
)

# 2. Get training images and ground truths address
all_img_paths = train_gen.input_img_paths + val_gen.input_img_paths
all_mask_paths = train_gen.target_img_paths + val_gen.target_img_paths

# 3. Random spilt initial labeled datas and unlabeled datas
combined = list(zip(all_img_paths, all_mask_paths))
random.seed(1337)
random.shuffle(combined)

initial_labeled_images = [x[0] for x in combined[:200]]   
initial_labeled_masks  = [x[1] for x in combined[:200]]
unlabeled_images       = [x[0] for x in combined[200:]]   
unlabeled_masks        = [x[1] for x in combined[200:]]

# 4. Active learning loop
active_learning_loop_ISIC(
    initial_labeled_images=initial_labeled_images,
    initial_labeled_masks=initial_labeled_masks,
    unlabeled_images=unlabeled_images,
    unlabeled_masks=unlabeled_masks,
    test_gen=test_gen,
    test_input_paths=test_input_paths,
    test_target_paths=test_target_paths,
    rounds=10,
    query_size=20,
    save_dir="D:/project/CODE/active_models_ISIC/",
    initial_model_path="D:/project/CODE/ISIC_segmentation.h5"  # load model
)
