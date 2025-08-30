import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class EntropySampling:
    def __init__(self, model, input_size=(160, 160)):
        self.model = model
        self.input_size = input_size

    def compute_entropy(self, prob_mask):
        entropy = -np.sum(prob_mask * np.log(prob_mask + 1e-10), axis=-1)
        return np.mean(entropy)

    def preprocess_image(self, path):
        img = load_img(path, target_size=self.input_size)
        img_array = img_to_array(img) / 255.0
        return img_array

    def query(self, unlabeled_images, n_query, verbose_every=200):
        scores = []
        total = len(unlabeled_images)

        for idx, path in enumerate(unlabeled_images):
            img_array = self.preprocess_image(path)
            pred = self.model.predict(img_array[None, ...], verbose=0)  
            entropy_score = self.compute_entropy(pred[0])
            scores.append(entropy_score)
            if (idx + 1) % verbose_every == 0 or (idx + 1) == total:
                print(f"Processed {idx + 1}/{total} images...")

        scores = np.array(scores)
        selected_indices = np.argsort(scores)[-n_query:]
        return selected_indices