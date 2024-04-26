"""
orb.py
This script is used to detect the keypoints and descriptors of an image using the ORB algorithm.

Authors: Alberto Castro Villasana , Ana Bárbara Quintero, Héctor Camacho Zamora
Organisation: UDEM
First created on Friday 23 April 2024
"""

# Importing the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """
    Parse command line arguments for image paths.
    
    Returns:
        Namespace: Parsed command line arguments with paths to the images.
    """
    parser = argparse.ArgumentParser(description='ORB feature matching between two images.')
    parser.add_argument('--image1', type=str, help='Path to the first image.')
    parser.add_argument('--image2', type=str, help='Path to the second image.')
    return parser.parse_args()

def load_and_resize_image(path, scale=0.4):
    """
    Load an image from a file and resize it.
    
    Args:
        path (str): Path to the image file.
        scale (float): Scaling factor for resizing the image.

    Returns:
        np.ndarray: The resized image.
    """
    image = cv2.imread(path) # Load the image
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.") # Raise an error if the image is not found
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale) # Resize the image
    return resized_image

def detect_features(image):
    """
    Detect and compute ORB features and descriptors in the image.
    
    Args:
        image (np.ndarray): Image in which to detect features.

    Returns:
        tuple: Keypoints and descriptors of the image.
    """
    orb = cv2.ORB_create(nfeatures=1000) # Create an ORB object, nfetures is the maximum number of features to retain
    keypoints, descriptors = orb.detectAndCompute(image, None) # Detect keypoints and compute descriptors
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Match ORB features using the Brute Force matcher.

    Args:
        desc1 (np.ndarray): Descriptors of the first image.
        desc2 (np.ndarray): Descriptors of the second image.

    Returns:
        list: Good matches after applying ratio test.
    """
    bf = cv2.BFMatcher() # Create a Brute Force matcher
    matches = bf.knnMatch(desc1, desc2, k=2) # Match descriptors of the two images
    good_matches = [] # List to store good matches
    for m, n in matches: # Apply ratio test
        if m.distance < 0.5 * n.distance: # If the distance is less than 0.5 times the next closest distance
            good_matches.append([m])
    return good_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    """
    Draw matches between two images.

    Args:
        image1 (np.ndarray): First image.
        keypoints1 (list): Keypoints in the first image.
        image2 (np.ndarray): Second image.
        keypoints2 (list): Keypoints in the second image.
        matches (list): Good matches to draw.

    Returns:
        np.ndarray: Image with drawn matches.
    """
    return cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, flags=2) # Draw matches

def display_images(images):
    """
    Display multiple images in separate windows.

    Args:
        images (dict): Dictionary of window names and image data.
    """
    for window_name, image in images.items(): # Display each image in a separate window
        cv2.imshow(window_name, image) 
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args() # Parse command line arguments
    image1 = load_and_resize_image(args.image1) # Load and resize the image
    image2 = load_and_resize_image(args.image2)
    keypoints1, descriptors1 = detect_features(image1) # Detect features in the images
    keypoints2, descriptors2 = detect_features(image2)
    good_matches = match_features(descriptors1, descriptors2) # Match features
    matched_image = draw_matches(image1, keypoints1, image2, keypoints2, good_matches) 
    display_images({'image1': image1, 'image2': image2, 'Matches': matched_image}) # Display the images