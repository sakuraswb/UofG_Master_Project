import numpy as np
from tensorflow.keras.preprocessing.image import load_img

def evaluate_iou_ISIC(test_preds, test_target_img_paths, num_classes=2, img_size=(160, 160), verbose_index=None):
    all_ious = []

    for i in range(len(test_preds)):
        pred_mask = (test_preds[i] > 0.5).astype(np.uint8).squeeze()
        gt_mask = load_img(test_target_img_paths[i], target_size=img_size, color_mode="grayscale")
        gt_mask = (np.array(gt_mask) > 127).astype(np.uint8)

        ious = []
        for cls in range(num_classes):
            pred_cls = (pred_mask == cls)
            gt_cls = (gt_mask == cls)
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            ious.append(np.nan if union == 0 else intersection / union)

        all_ious.append(ious)

    all_ious = np.array(all_ious)
    mean_iou_all = np.nanmean(np.nanmean(all_ious, axis=1))  # average IoU
    mean_per_class = np.nanmean(all_ious, axis=0)            # average IoU of each class

    print(f"Mean IoU (all images): {mean_iou_all:.4f}")
    print("Per-class IoU:", ["{:.4f}".format(x) for x in mean_per_class])

    if verbose_index is not None:
        ious = all_ious[verbose_index]
        mean_iou_single = np.nanmean(ious)
        print(f"\n[Image {verbose_index}] Mean IoU: {mean_iou_single:.4f}")
        for cls, iou in enumerate(ious):
            print(f"Class {cls} IoU: {iou:.4f}")

    return mean_iou_all, mean_per_class, all_ious



def evaluate_ssim_scores_ISIC(test_preds, test_target_img_paths, img_size=(160, 160), verbose_index=None):
    from skimage.metrics import structural_similarity as ssim
    ssim_scores = []

    for i in range(len(test_preds)):
        pred_mask = (test_preds[i] > 0.5).astype(np.uint8).squeeze()
        gt_mask = load_img(test_target_img_paths[i], target_size=img_size, color_mode="grayscale")
        gt_mask = (np.array(gt_mask) > 127).astype(np.uint8)

        gt = gt_mask.squeeze().astype(np.uint8)
        pred = pred_mask.squeeze().astype(np.uint8)

        data_range = max(gt.max(), pred.max()) - min(gt.min(), pred.min())
        if data_range == 0:
            score = 1.0 if np.array_equal(gt, pred) else 0.0
        else:
            score = ssim(gt, pred, data_range=data_range)

        ssim_scores.append(score)

    mean_ssim = np.mean(ssim_scores)
    print(f"Mean SSIM over validation set: {mean_ssim:.4f}")

    if verbose_index is not None:
        print(f"[Image {verbose_index}] SSIM: {ssim_scores[verbose_index]:.4f}")

    return mean_ssim, ssim_scores
