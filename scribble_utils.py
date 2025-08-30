import numpy as np
from PIL import Image

def convert_scribble_to_pseudo_mask(scribble_path, target_size=(160, 160)):
#Convert the user-annotated scribble image into a pseudo-mask. Green (0,255,0) → Class 2 (Pet)  Other → Class 0 (Background)
    img = Image.open(scribble_path).convert("RGB").resize(target_size)
    arr = np.array(img)

    mask = np.zeros(target_size, dtype=np.uint8)
    green = (0, 255, 0)
    mask[(arr == green).all(axis=-1)] = 2
    return mask
