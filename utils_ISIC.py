import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img

def create_mask(pred):
    return np.argmax(pred, axis=-1)[..., np.newaxis]

def display_sample_ISIC(test_preds, test_input_img_paths, test_target_img_paths, 
                            all_ious, ssim_scores, round_idx, sample_indices, 
                            base_save_dir="D:/project/CODE/predicted_image_ISIC/", 
                            img_size=(160,160)):

    round_dir = os.path.join(base_save_dir, f"round_{round_idx}")
    os.makedirs(round_dir, exist_ok=True)

    for idx in sample_indices:
        input_img = load_img(test_input_img_paths[idx], target_size=img_size)
        gt_gray = load_img(test_target_img_paths[idx], target_size=img_size, color_mode="grayscale")
        gt_mask_arr = np.array(gt_gray).astype("uint8")
        gt_mask_arr = (gt_mask_arr > 127).astype("uint8") 
        gt_mask_arr = np.expand_dims(gt_mask_arr, axis=-1)
        gt_mask = Image.fromarray((gt_mask_arr.squeeze() * 255).astype(np.uint8))
        pred_mask = (test_preds[idx] > 0.5).astype(np.uint8)  
        pred_img = Image.fromarray((pred_mask.squeeze() * 255).astype(np.uint8))

        ious = all_ious[idx]
        mean_iou = np.nanmean(ious)
        ssim_score = ssim_scores[idx]

        plt.figure(figsize=(12, 4))
        titles = ["Input Image", "Ground Truth", "Predicted Mask"]
        images = [input_img, gt_mask, pred_img]

        for j in range(3):
            plt.subplot(1, 3, j + 1)
            plt.title(titles[j])
            plt.imshow(images[j])
            plt.axis("off")

        plt.figtext(0.5, 0.01,
            f"Mean IoU: {mean_iou:.4f} | Class 0: {ious[0]:.4f} | Class 1: {ious[1]:.4f} | SSIM: {ssim_score:.4f}",
            ha="center", fontsize=10, wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        filename = os.path.basename(test_input_img_paths[idx])
        name_no_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(round_dir, f"{idx:04d}_{name_no_ext}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def export_metrics_to_csv_ISIC(test_input_img_paths, all_ious, ssim_scores, 
                                  output_path="D:/project/CODE/predicted_image_ISIC/metrics.csv"):
    rows = []

    for i in range(len(all_ious)):
        filename = os.path.basename(test_input_img_paths[i])
        ious = all_ious[i]
        mean_iou = np.nanmean(ious)
        ssim_score = ssim_scores[i]

        row = {
            "index": i,
            "filename": filename,
            "mean_iou": mean_iou,
            "class_0_iou": ious[0],
            "class_1_iou": ious[1],
            "ssim": ssim_score
        }
        rows.append(row)

    all_ious_array = np.array(all_ious)
    mean_all = {
        "index": "mean",
        "filename": "ALL",
        "mean_iou": np.nanmean(np.nanmean(all_ious_array, axis=1)),
        "class_0_iou": np.nanmean(all_ious_array[:, 0]),
        "class_1_iou": np.nanmean(all_ious_array[:, 1]),
        "ssim": np.mean(ssim_scores)
    }
    rows.append(mean_all)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
