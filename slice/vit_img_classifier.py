import tensorflow as tf
import numpy as np
from transformers import ViTFeatureExtractor, TFViTForImageClassification

class ViTImageClassifier:
    def __init__(self, model_name):
        self.name = "vitp16"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = TFViTForImageClassification.from_pretrained(model_name)

    def preprocess_image(self, images):
        #image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        #inputs = self.feature_extractor(images=images, return_tensors="tf")
        images = images / 255.0

        return images


    def predict_old(self, inputs, verbose=0):
        outputs = self.model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()
        top_probability = tf.reduce_max(probabilities).numpy()
        return predicted_class, top_probability

    def predict(self, images, verbose=0): # this extracts and also predicts
        # Preprocess the batch of images
        images = np.round(images)
        inputs = self.feature_extractor(images=images, return_tensors="tf")

        # Make predictions using the pre-trained model
        outputs = self.model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)

        if verbose > 0:
            predicted_classes = tf.argmax(probabilities, axis=-1).numpy()
            top_probabilities = tf.reduce_max(probabilities, axis=-1).numpy()
            for i in range(len(images)):
                print(
                    f"Image {i + 1}: Predicted class index: {predicted_classes[i]} with probability: {top_probabilities[i]}")

        return probabilities.numpy()

    def model_name(self):
        return self.name
