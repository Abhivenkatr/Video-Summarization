import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the trained model
model_path = 'hyperkvasir_model.h5'

# Directory containing the unlabeled images
unlabeled_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-unlabeled-images\unlabeled-images\images"

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Ensure the unlabeled directory exists
assert os.path.exists(unlabeled_dir) and len(os.listdir(unlabeled_dir)) > 0, "Unlabeled directory is empty or does not exist."

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to load and predict classes for unlabeled images
def predict_unlabeled_images(unlabeled_dir):
    predictions = []
    for img_file in os.listdir(unlabeled_dir):
        img_path = os.path.join(unlabeled_dir, img_file)
        img = preprocess_image(img_path)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_label = class_indices[class_index]
        predictions.append((img_file, class_label))
        print(f"Processed {img_file} - Predicted Class: {class_label}")
    return predictions

# Get class indices from the training data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\flat_structure\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
class_indices = {v: k for k, v in train_generator.class_indices.items()}

# Predict classes for unlabeled images
predictions = predict_unlabeled_images(unlabeled_dir)

# Print or save the predictions
output_file = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-unlabeled-images\unlabeled-images\predictions.txt"
with open(output_file, 'w') as f:
    for img_file, class_label in predictions:
        f.write(f"Image: {img_file} - Predicted Class: {class_label}\n")
        print(f"Image: {img_file} - Predicted Class: {class_label}")
print(f"Predictions saved to {output_file}")
