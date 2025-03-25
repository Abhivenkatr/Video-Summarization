import os
import csv
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('hyperkvasir_model.h5')

# Data paths
video_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-videos\hyper-kvasir-videos\videos"
annotations_file = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-videos\hyper-kvasir-videos\video_annotations.csv"
metadata_file = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-videos\hyper-kvasir-videos\video_summaries_metadata.csv"
output_summary_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-videos\hyper-kvasir-videos\summarized_videos"

# Ensure directories and files are valid
assert os.path.exists(video_dir) and len(os.listdir(video_dir)) > 0, "Video directory is empty or does not exist."
assert os.path.exists(annotations_file), "Annotations file does not exist."
if not os.path.exists(output_summary_dir):
    os.makedirs(output_summary_dir)

# Function to read video annotations
def read_annotations(file_path):
    annotations = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames
        print(f"CSV Headers: {headers}")  # Print headers for debugging
        for row in reader:
            print(f"Row: {row}")  # Print row for debugging
            video_id = row.get('videoID', '').strip()
            finding = row.get('finding', '').strip()
            if video_id:
                annotations[video_id] = finding
    return annotations

# Load annotations
annotations = read_annotations(annotations_file)
print(f"Annotations: {annotations}")  # Debug: print loaded annotations

# Function to extract keyframes
def extract_keyframes(video_path):
    keyframes = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keyframes.append(frame)
    cap.release()
    return keyframes

# Load class labels
train_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\flat_structure\train"
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Function to map prediction to one of the class labels
def get_class_label(prediction, class_labels):
    class_idx = np.argmax(prediction)
    return class_labels[class_idx]

# Video summarization and metadata collection
def summarize_video(video_dir, annotations, model, metadata_file, output_summary_dir):
    with open(metadata_file, mode='w', newline='') as metadata_csv:
        fieldnames = ['video_id', 'finding', 'summary']
        writer = csv.DictWriter(metadata_csv, fieldnames=fieldnames)
        writer.writeheader()

        for video_file in os.listdir(video_dir):
            video_id, _ = os.path.splitext(video_file)
            if video_id in annotations:
                video_path = os.path.join(video_dir, video_file)
                print(f"Processing video: {video_path}")  # Debug: print video path
                keyframes = extract_keyframes(video_path)

                abnormal_detected = False
                abnormal_class = ""

                for frame in keyframes:
                    frame_resized = cv2.resize(frame, (224, 224))
                    frame_array = np.expand_dims(frame_resized, axis=0) / 255.0
                    prediction = model.predict(frame_array)[0]  # Get the prediction for the single frame
                    print(f"Prediction: {prediction}")  # Debug: print prediction

                    if prediction.max() > 0.5:
                        abnormal_detected = True
                        abnormal_class = get_class_label(prediction, class_labels)
                        print(f"Abnormal class detected: {abnormal_class}")  # Debug: print abnormal class
                        break

                if abnormal_detected:
                    summary_description = f"Video annotation: {annotations[video_id]}, Class summary: Abnormal ({abnormal_class})"
                else:
                    summary_description = f"Video annotation: {annotations[video_id]}, Class summary: Normal"

                # Write metadata
                writer.writerow({
                    'video_id': video_id,
                    'finding': annotations[video_id],
                    'summary': summary_description
                })
                print(f"Metadata written for video: {video_id}")  # Debug: confirm metadata write

                # Save summary video (if any frames are selected as summary frames)
                if abnormal_detected:
                    output_video_path = os.path.join(output_summary_dir, f"{video_id}_summary.mp4")
                    height, width, _ = keyframes[0].shape
                    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                    for frame in keyframes:
                        out.write(frame)
                    out.release()
                    print(f"Summary video saved for: {video_id}")  # Debug: confirm summary video save

    print("Video summarization and metadata creation complete.")

# Perform video summarization and metadata collection
summarize_video(video_dir, annotations, model, metadata_file, output_summary_dir)
