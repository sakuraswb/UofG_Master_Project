import numpy as np
def predict_masks_ISIC(model, data_gen, threshold=0.5):
    preds = model.predict(data_gen)
    preds_bin = (preds > threshold).astype(np.uint8)
    return preds_bin
