import os
import time
import random
import gc
import tensorflow as tf
from model import get_unet_model
from train import train_model
from predict import predict_masks
from eval import evaluate_iou, evaluate_ssim_scores
from data_loader import OxfordPets
from tensorflow.keras.models import load_model
from query_strategies.entropy_sampling import EntropySampling
from query_strategies.margin_sampling import MarginSampling
from query_strategies.random_sampling import RandomSampling
from utils import display_sample, export_metrics_to_csv
import numpy as np


def active_learning_loop(
    initial_labeled_images,
    initial_labeled_masks,
    unlabeled_images,
    unlabeled_masks,
    test_gen,
    test_input_paths,            
    test_target_paths,
    rounds=5,
    query_size=100,
    save_dir="D:/project/CODE/active_models/",
    initial_model_path="D:/project/CODE/oxford_segmentation.h5"
):
    img_size = (160, 160)
    batch_size = 8
    os.makedirs(save_dir, exist_ok=True)
    fixed_sample_indices = None

    for round in range(rounds):
        print(f"\n======== Round {round+1}/{rounds} ========")

        # === 1. load or train model ===
        
        t0 = time.time()
        if round == 0:
            if os.path.exists(initial_model_path):
                print(f"Loading initial pretrained model: {initial_model_path}")                
                model = load_model(initial_model_path, compile=False)                
            else:
                print("No pretrained model found. Building a new model...")
                model = get_unet_model(input_shape=(160, 160, 3), num_classes=3)
        else:
            prev_model_path = os.path.join(save_dir, f"model_round_{round}.h5")
            print(f"Loading model from previous round: {prev_model_path}")
            model = load_model(prev_model_path, compile=False)
        
        print(f"load model cost: {time.time() - t0:.2f} seconds")


        # === 2. load dataset ===        
        train_gen = OxfordPets(batch_size, img_size, initial_labeled_images, initial_labeled_masks)       

        # === 3. train and save model ===
        t0 = time.time()
        current_model_path = os.path.join(save_dir, f"model_round_{round+1}.h5")
        print(f"Training model and saving to: {current_model_path}")
        train_model(model, train_gen, train_gen, save_path=current_model_path)
        print(f"train cost: {time.time() - t0:.2f} seconds")

        # === 4. AL sampling ===
        t0 = time.time()
        print("Selecting uncertain samples using RandomSampling...")
        strategy = RandomSampling(model)
        selected_indices = strategy.query(unlabeled_images, query_size)
        print(f"sampling cost: {time.time() - t0:.2f} seconds")

        # === 5. annotating and update labeled pool ===
        t0 = time.time()
        new_imgs = [unlabeled_images[i] for i in selected_indices]
        new_masks = [unlabeled_masks[i] for i in selected_indices]
        initial_labeled_images += new_imgs
        initial_labeled_masks += new_masks
        for i in sorted(selected_indices, reverse=True):
            del unlabeled_images[i]
            del unlabeled_masks[i]
        print(f"update cost: {time.time() - t0:.2f} seconds")

        # === 6. predicting ===
        t0 = time.time()
        print("Predicting and evaluating model on test set...")
        test_preds = predict_masks(model, test_gen)
        print(f"predict cost: {time.time() - t0:.2f} seconds")

        if fixed_sample_indices is None:
            random.seed(1337)
            total_images = len(test_preds)
            fixed_sample_indices = random.sample(range(total_images), min(100, total_images))
            

        # metrics
        mean_iou, per_class_iou, all_ious = evaluate_iou(test_preds, test_target_paths)               
        mean_ssim, ssim_scores = evaluate_ssim_scores(test_preds, test_target_paths)
        

        # save results
        t0 = time.time()
        csv_path = os.path.join(save_dir, f"metrics_round_{round+1}.csv")
        export_metrics_to_csv(
            test_input_img_paths=test_input_paths,
            all_ious=all_ious,
            ssim_scores=ssim_scores,
            output_path=csv_path
        )
        print(f"save cost: {time.time() - t0:.2f} seconds")
        display_sample(
            test_preds=test_preds,
            test_input_img_paths=test_input_paths,
            test_target_img_paths=test_target_paths,
            all_ious=all_ious,
            ssim_scores=ssim_scores,
            round_idx=round+1,
            sample_indices=fixed_sample_indices,
            base_save_dir="D:/project/CODE/predicted_image/"
        )

        print(" Round complete.\n")
        
        del model
        del test_preds
        gc.collect()
        tf.keras.backend.clear_session()