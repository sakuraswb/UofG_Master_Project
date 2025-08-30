import os
import numpy as np
import shutil  
import pandas as pd 
import random
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import get_unet_model
from train import train_model
from predict import predict_masks
from eval import evaluate_iou, evaluate_ssim_scores
from data_loader import OxfordPets
from query_strategies.entropy_sampling import EntropySampling
from query_strategies.margin_sampling import MarginSampling
from query_strategies.random_sampling import RandomSampling
from utils import display_sample, export_metrics_to_csv
from scribble_utils import convert_scribble_to_pseudo_mask

def active_learning_loop_scribble(
    initial_labeled_images,
    initial_labeled_masks,
    unlabeled_images,
    test_gen,
    test_input_paths,
    test_target_paths,
    scribble_dir="D:/project/CODE/scribbles/",
    rounds=5,
    query_size=30,
    save_dir="D:/project/CODE/scribble_models/",
    initial_model_path="D:/project/CODE/oxford_segmentation.h5"
):
    img_size = (160, 160)
    batch_size = 8
    os.makedirs(save_dir, exist_ok=True)    
    fixed_sample_indices = None

    for round in range(rounds):
        print(f"\n======== Round {round+1}/{rounds} ========")

        # === 1. load or train model ===
        if round == 0:
            if os.path.exists(initial_model_path):
                print(f"Loading initial pretrained model: {initial_model_path}")
                model = load_model(initial_model_path, compile=False)
            else:
                print("No pretrained model found. Building and training a new model...")
                model = get_unet_model(input_shape=(160, 160, 3), num_classes=3)
                train_gen_init = OxfordPets(batch_size, img_size, initial_labeled_images, initial_labeled_masks)
                train_model(model, train_gen_init, train_gen_init, save_path=initial_model_path)
                print(f"Initial model trained and saved to: {initial_model_path}")
        else:
            prev_model_path = os.path.join(save_dir, f"model_round_{round}.h5")
            print(f"Loading model from previous round: {prev_model_path}")
            model = load_model(prev_model_path, compile=False)

        # === 2. load dataset ===
        train_gen = OxfordPets(batch_size, img_size, initial_labeled_images, initial_labeled_masks)

        # === 3. train and save model ===
        current_model_path = os.path.join(save_dir, f"model_round_{round+1}.h5")
        print(f"Training model and saving to: {current_model_path}")
        train_model(model, train_gen, train_gen, save_path=current_model_path)

        # === 4. AL sampling ===
        print("Selecting uncertain samples using MarginSampling...")
        strategy = MarginSampling(model)
        selected_indices = strategy.query(unlabeled_images, query_size)
        selected_paths = [unlabeled_images[i] for i in selected_indices]

        # === 5. create scribble list ===
        round_scribble_dir = os.path.join(scribble_dir, f"round_{round+1}")
        os.makedirs(round_scribble_dir, exist_ok=True)
        print(f"\n please scribble  {len(selected_paths)} pictures below，save to：{round_scribble_dir}")
        for p in selected_paths:
            fname = os.path.basename(p)
            dst = os.path.join(round_scribble_dir, fname)
            if not os.path.exists(dst):
                shutil.copy(p, dst)
        input("when scribble finished，press Enter to continue...")

        # === 6. save scribble list ===
        sample_csv_path = os.path.join(save_dir, f"round_{round+1}_to_label.csv")
        pd.DataFrame({"image_name": [os.path.basename(p) for p in selected_paths]}).to_csv(sample_csv_path, index=False)

        # === 7. load scribble ===
        new_imgs, new_masks = [], []
        for p in selected_paths:
            fname = os.path.basename(p)
            scribble_path = os.path.join(round_scribble_dir, fname)
            if not os.path.exists(scribble_path):
                print(f" scribble not exist：{scribble_path}")
                continue
            new_imgs.append(p)
            new_masks.append(convert_scribble_to_pseudo_mask(scribble_path, target_size=img_size))

        # === 8. update labeled pool ===
        initial_labeled_images += new_imgs
        initial_labeled_masks += new_masks
        for i in sorted(selected_indices, reverse=True):
            del unlabeled_images[i]

        # === 9. metric ===
        print("Predicting and evaluating model on test set...")
        test_preds = predict_masks(model, test_gen)
        mean_iou, per_class_iou, all_ious = evaluate_iou(test_preds, test_target_paths)
        mean_ssim, ssim_scores = evaluate_ssim_scores(test_preds, test_target_paths)

        if fixed_sample_indices is None:
            random.seed(1337)
            fixed_sample_indices = random.sample(range(len(test_preds)), min(100, len(test_preds)))

        # === 10. save result ===
        export_metrics_to_csv(test_input_paths, all_ious, ssim_scores,
                              output_path=os.path.join(save_dir, f"metrics_round_{round+1}.csv"))

        display_sample(test_preds, test_input_paths, test_target_paths,
                       all_ious, ssim_scores, round_idx=round+1,
                       sample_indices=fixed_sample_indices,
                       base_save_dir="D:/project/CODE/predicted_image_scribble/")

        # clear memory
        del model, test_preds
        gc.collect()
        tf.keras.backend.clear_session()

        print(f" Round {round+1} completed.\n")
