import cv2
import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define the folder path as a variable
dataset_folder_path = '/Users/mac/Desktop/RealtimeFaceRecognition/dataset'

def load_images_from_folder(folder_path, img_size=(128, 128)):
    images = []       # To store the processed images
    labels = []       # To store corresponding labels (names)

    # Iterate through each subfolder (each representing a person)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        # Ensure the path is a directory
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                
                # Load and preprocess the image
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize the image to a fixed size (e.g., 64x64 pixels)
                    img = cv2.resize(img, img_size)
                    
                    # Convert BGR (OpenCV default) to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Append the processed image and the label
                    images.append(img)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_and_save(folder_path, output_path='data.pkl', img_size=(128, 128)):
    # Load images and labels
    images, labels = load_images_from_folder(folder_path, img_size)
    
    # Normalize the images to range [0, 1]
    images = images / 255.0

    # Encode labels (names) to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Save preprocessed data
    with open(output_path, 'wb') as f:
        pickle.dump((x_train, x_test, y_train, y_test, le), f)
    
    print("Data preprocessed and saved to", output_path)

def check_labels_and_counts(folder_path):
    label_counts = {}
    
    # Iterate through each subfolder (each representing a person)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        # Ensure the path is a directory
        if os.path.isdir(label_path):
            # Count the number of images in the subfolder
            num_images = len([f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))])
            label_counts[label] = num_images
    
    return label_counts

preprocess_and_save(dataset_folder_path)

# Check and print labels and their image counts
label_counts = check_labels_and_counts(dataset_folder_path)
print("Labels and their image counts:")
for label, count in label_counts.items():
    print(f"{label}: {count} images")