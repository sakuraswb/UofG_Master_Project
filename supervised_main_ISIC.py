import os
from model_ISIC import get_unet_model_ISIC
from data_loader_ISIC import split_dataset
from train_ISIC import train_model_ISIC
from predict_ISIC import predict_masks_ISIC
from tensorflow.keras.models import load_model
from eval_ISIC import evaluate_iou_ISIC, evaluate_ssim_scores_ISIC
from utils_ISIC import display_sample_ISIC, export_metrics_to_csv_ISIC
import numpy as np
import matplotlib.pyplot as plt

# data path
input_dir = "D:/project/CODE/ISIC2017/Training_Data/"
target_dir = "D:/project/CODE/ISIC2017/Ground_Truth/"
model_path = "D:/project/CODE/ISIC_segmentation.h5"  

# load dataset
train_gen, val_gen, test_gen, test_input_paths, test_target_paths = split_dataset(
    input_dir,
    target_dir,
    img_size=(160, 160),
    batch_size=8,
    max_images=1000,
    save_split_dir="splits_ISIC_2000"
)

if os.path.exists(model_path):
    print("Loading pretrained ISIC model")
    model = load_model(model_path, compile=False)
    history = None
else:
    print("Training new ISIC model")
    model = get_unet_model_ISIC(input_shape=(160,160,3), num_classes=1)
    history = train_model_ISIC(model, train_gen, val_gen, save_path=model_path)

# show training result
sample_img, sample_mask = train_gen[0]  
pred = model.predict(sample_img)
pred_bin = (pred > 0.5).astype(np.uint8)

plt.subplot(1,3,1)
plt.imshow(sample_img[0])
plt.title("Train Input")
plt.subplot(1,3,2)
plt.imshow(sample_mask[0].squeeze(), cmap='gray')
plt.title("Ground Truth")
plt.subplot(1,3,3)
plt.imshow(pred_bin[0].squeeze(), cmap='gray')
plt.title("Predicted Mask")
plt.show()

# predict
test_preds = predict_masks_ISIC(model, test_gen, threshold=0.5)

plt.imshow(test_preds[0].squeeze(), cmap='gray')  
plt.title("Probability Map")
plt.colorbar()
plt.show()

pred_bin = (test_preds[0] > 0.5).astype(np.uint8)
plt.imshow(pred_bin.squeeze(), cmap='gray')  
plt.title("Binary Mask (>0.5)")
plt.show()
# calculate metrics
mean_iou, per_class_iou, all_ious = evaluate_iou_ISIC(test_preds, test_target_paths)
mean_ssim, ssim_scores = evaluate_ssim_scores_ISIC(test_preds, test_target_paths)

display_sample_ISIC(
    test_preds=test_preds,
    test_input_img_paths=test_input_paths,
    test_target_img_paths=test_target_paths,
    all_ious=all_ious,
    ssim_scores=ssim_scores,
    round_idx=1,           
    sample_indices=[0]     
)


# save result to csv
export_metrics_to_csv_ISIC(
    test_input_img_paths=test_input_paths,
    all_ious=all_ious,
    ssim_scores=ssim_scores,
    output_path="D:/project/CODE/predicted_image_ISIC/metrics_ISIC.csv"
)


