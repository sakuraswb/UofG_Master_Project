import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

class RandomSampling:
    def __init__(self, model=None, input_size=(160, 160), seed=121):
        self.model = model
        self.input_size = input_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def preprocess_image(self, path):
        img = load_img(path, target_size=self.input_size)
        img_array = img_to_array(img) / 255.0
        return img_array

    def query(self, unlabeled_images, n_query, verbose_every=200):
        total = len(unlabeled_images)
        if total <= n_query:
            return list(range(total)) 

        selected_indices = random.sample(range(total), n_query)
        return selected_indices
