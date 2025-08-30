import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageOps , Image
from tensorflow.keras.preprocessing.image import load_img, array_to_img

CLASS2COLOR = {
    0: (0, 128, 128),     # background
    1: (255, 255, 0),     # edge
    2: (128, 0, 128)      # mainbody
}

def class2rgb(class_mask, class2color):
# convert to RGB images
    if class_mask.ndim == 3:
        class_mask = class_mask[..., 0]
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in class2color.items():
        rgb[class_mask == cls] = color
    return rgb

def create_mask(pred):
    return np.argmax(pred, axis=-1)[..., np.newaxis]  # shape: (H, W, 1)

def display_sample(test_preds, test_input_img_paths, test_target_img_paths, 
                         all_ious, ssim_scores, round_idx, sample_indices, 
                         base_save_dir="D:/project/CODE/predicted_image/", img_size=(160,160)):

    round_dir = os.path.join(base_save_dir, f"round_{round_idx}")
    os.makedirs(round_dir, exist_ok=True)

    for idx in sample_indices:
        # Input images
        input_img = load_img(test_input_img_paths[idx], target_size=img_size)

        # Ground truth
        gt_gray = load_img(test_target_img_paths[idx], target_size=img_size, color_mode="grayscale")
        gt_mask_arr = np.array(gt_gray).astype("uint8") - 1
        gt_mask_arr = np.expand_dims(gt_mask_arr, axis=-1)
        gt_rgb_array = class2rgb(gt_mask_arr, CLASS2COLOR)
        gt_mask = Image.fromarray(gt_rgb_array)

        # Predicted mask
        pred_mask = np.argmax(test_preds[idx], axis=-1)[..., np.newaxis]
        pred_rgb_array = class2rgb(pred_mask, CLASS2COLOR)
        pred_img = Image.fromarray(pred_rgb_array)

        # Metrics
        ious = all_ious[idx]
        mean_iou = np.nanmean(ious)
        ssim_score = ssim_scores[idx]

        # Show
        plt.figure(figsize=(12, 4))
        titles = ["Input Image", "Ground Truth", "Predicted Mask"]
        images = [input_img, gt_mask, pred_img]

        for j in range(3):
            plt.subplot(1, 3, j + 1)
            plt.title(titles[j])
            plt.imshow(images[j])
            plt.axis("off")

        plt.figtext(0.5, 0.01,
            f"Mean IoU: {mean_iou:.4f} | Class 0: {ious[0]:.4f} | Class 1: {ious[1]:.4f} | Class 2: {ious[2]:.4f} | SSIM: {ssim_score:.4f}",
            ha="center", fontsize=10, wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        filename = os.path.basename(test_input_img_paths[idx])
        name_no_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(round_dir, f"{idx:04d}_{name_no_ext}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def export_metrics_to_csv(test_input_img_paths, all_ious, ssim_scores, output_path="D:/project/CODE/predicted_image/metrics.csv"):
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
            "class_2_iou": ious[2],
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
        "class_2_iou": np.nanmean(all_ious_array[:, 2]),
        "ssim": np.mean(ssim_scores)
    }
    rows.append(mean_all)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
