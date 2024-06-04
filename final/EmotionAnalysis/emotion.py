import cv2
import numpy as np
from deepface.commons import functions, distance as dst

from EmotionAnalysis.model import loadModel

# Initializing the model globally
model = loadModel()

def analysis(image_list):
    # Detect face, extract detected face, grayscale transformation, and vectorize
    vectorized_images = []
    img_indices = []  # To keep track of indices of images that contain faces
    for i, img in enumerate(image_list):
        gray_face = functions.preprocess_face(img, target_size = (48, 48), grayscale = True, enforce_detection=False)
        if gray_face is None:  # No face detected, continue to the next image
            continue
        gray_face = np.expand_dims(gray_face, axis=0)
        embedding = model.predict(gray_face)[0,:]
        vectorized_images.append(embedding)
        img_indices.append(i)  # Save the index of the image that contains a face

    # Calculate the average feature vector
    avg_vector = np.mean(vectorized_images, axis=0)

    # Calculate the cosine distance between each face and the average
    distances = []
    for i, embedding in enumerate(vectorized_images):
        distance = dst.findCosineDistance(avg_vector, embedding)
        distances.append((distance, img_indices[i]))  # Use the original image index

    # Find the face with the maximum distance from the average
    max_distance, idx = max(distances, key=lambda x: x[0])

    # Return the index of the face that is the farthest from the average
    return idx
