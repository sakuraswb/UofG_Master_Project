import os
import random
import numpy as np
import gc
import tensorflow as tf
from model_ISIC import get_unet_model_ISIC
from train_ISIC import train_model_ISIC
from predict_ISIC import predict_masks_ISIC
from eval_ISIC import evaluate_iou_ISIC, evaluate_ssim_scores_ISIC
from data_loader_ISIC import ISICDataset
from tensorflow.keras.models import load_model
from query_strategies.entropy_sampling import EntropySampling
from query_strategies.margin_sampling import MarginSampling
from query_strategies.random_sampling import RandomSampling
from utils_ISIC import display_sample_ISIC, export_metrics_to_csv_ISIC

def active_learning_loop_ISIC(
    initial_labeled_images,
    initial_labeled_masks,
    unlabeled_images,
    unlabeled_masks,
    test_gen,
    test_input_paths,            
    test_target_paths,
    rounds=5,
    query_size=100,
    save_dir="D:/project/CODE/active_models_ISIC/",
    initial_model_path="D:/project/CODE/ISIC_segmentation.h5"
):
    img_size = (160, 160)
    batch_size = 8
    os.makedirs(save_dir, exist_ok=True)


    fixed_sample_indices = None

    for round in range(rounds):
        print(f"\n======== Round {round+1}/{rounds} ========")

        # === 1. load or train model ===
        if round == 0:
            print(f"Loading initial pretrained model: {initial_model_path}")
            model = load_model(initial_model_path, compile=False)
        else:
            prev_model_path = os.path.join(save_dir, f"model_round_{round}.h5")
            print(f"Loading model from previous round: {prev_model_path}")
            model = load_model(prev_model_path, compile=False)

        # === 2. load dataset ===
        train_gen = ISICDataset(batch_size, img_size, initial_labeled_images, initial_labeled_masks)

        # === 3. train and save model ===
        current_model_path = os.path.join(save_dir, f"model_round_{round+1}.h5")
        print(f"Training model and saving to: {current_model_path}")
        train_model_ISIC(model, train_gen, train_gen, save_path=current_model_path)

        # === 4. AL sampling===
        print("Selecting uncertain samples using EntropySampling...")
        strategy = EntropySampling(model)
        selected_indices = strategy.query(unlabeled_images, query_size)

        # === 5. annotationg and update labeled pool ===
        new_imgs = [unlabeled_images[i] for i in selected_indices]
        new_masks = [unlabeled_masks[i] for i in selected_indices]
        initial_labeled_images += new_imgs
        initial_labeled_masks += new_masks

        for i in sorted(selected_indices, reverse=True):
            del unlabeled_images[i]
            del unlabeled_masks[i]

       
        print("Predicting on test set...")
        test_preds = predict_masks_ISIC(model, test_gen)

        if fixed_sample_indices is None:
            random.seed(1337)
            total_images = len(test_preds)
            fixed_sample_indices = random.sample(range(total_images), min(100, total_images))
            print(f" totally {len(fixed_sample_indices)} images")

        #  metrics 
        mean_iou, per_class_iou, all_ious = evaluate_iou_ISIC(test_preds, test_target_paths)
        mean_ssim, ssim_scores = evaluate_ssim_scores_ISIC(test_preds, test_target_paths)
        csv_path = os.path.join(save_dir, f"metrics_round_{round+1}.csv")
        export_metrics_to_csv_ISIC(
            test_input_img_paths=test_input_paths,
            all_ious=all_ious,
            ssim_scores=ssim_scores,
            output_path=csv_path
        )

        # save result
        display_sample_ISIC(
            test_preds=test_preds,
            test_input_img_paths=test_input_paths,
            test_target_img_paths=test_target_paths,
            all_ious=all_ious,
            ssim_scores=ssim_scores,
            round_idx=round+1,
            sample_indices=fixed_sample_indices,
            base_save_dir="D:/project/CODE/predicted_image_ISIC/"
        )

        # clear memory
        del model
        del test_preds
        gc.collect()
        tf.keras.backend.clear_session()

        print(f" Round {round+1} completed.\n")
