import os
from model import get_unet_model
from data_loader import split_dataset
from train import train_model
from predict import predict_masks
from tensorflow.keras.models import load_model
from eval import evaluate_iou, evaluate_ssim_scores
from utils import display_sample, export_metrics_to_csv

# data path
input_dir = "D:/project/CODE/oxford_pet_data/oxford-iiit-pet/images/"
target_dir = "D:/project/CODE/oxford_pet_data/oxford-iiit-pet/annotations/trimaps/"
model_path = "D:/project/CODE/oxford_segmentation.h5"

# load dataset
train_gen, val_gen, test_gen, test_input_paths, test_target_paths = split_dataset(
    input_dir,
    target_dir,
    img_size=(160,160),   
    batch_size=4,         
    max_images=1000,
    save_split_dir="splits_1000"      
)


if os.path.exists(model_path):
    print("loading pretrained model")
    model = load_model(model_path, compile=False)
    history = None
else:
    print("training new model")
    model = get_unet_model()
    history = train_model(model, train_gen, val_gen, save_path=model_path)

test_preds = predict_masks(model, test_gen)

mean_iou, per_class_iou, all_ious = evaluate_iou(test_preds, test_target_paths)
mean_ssim, ssim_scores = evaluate_ssim_scores(test_preds, test_target_paths)

display_sample(
    i=136,
    test_preds=test_preds,
    test_input_img_paths=test_input_paths,
    test_target_img_paths=test_target_paths,
    all_ious=all_ious,
    ssim_scores=ssim_scores
)

export_metrics_to_csv(
    test_input_img_paths=test_input_paths,
    all_ious=all_ious,
    ssim_scores=ssim_scores
)