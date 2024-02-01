
import cv2
import numpy as np
from skimage.feature import hog
import os

# Preprocessing for road detection in satellite images
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    # Resize the image to the target size
    img = cv2.resize(img, target_size)
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Apply adaptive thresholding to create a binary image
    _, thresholded_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_img

def enhance_image(image):
    # Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)
    return enhanced_img

# Feature extraction using HOG for road detection
def extract_features(image):
    # Extract features using Histogram of Oriented Gradients (HOG) for road detection
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features, hog_image

# Saving preprocessed images
def save_preprocessed_image(image, output_path):
    cv2.imwrite(output_path, image)

# Example usage for multiple images with resizing
input_image_directory = 'data/GGPL'
output_directory = 'preprocessed'
target_size = (1700, 800)  # Specify the target size for resizing

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each image in the input directory
for filename in os.listdir(input_image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
        input_image_path = os.path.join(input_image_directory, filename)
        # Preprocess, resize, enhance, extract features, and save the image
        image = preprocess_image(input_image_path, target_size)
        enhanced_image = enhance_image(image)  # Assuming enhance_image function is defined
        features, hog_image = extract_features(enhanced_image)  # Assuming extract_features function is defined
        output_image_path = os.path.join(output_directory, 'preprocessed_' + filename)
        save_preprocessed_image(enhanced_image, output_image_path)
