from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine

from model import *

model = loadModel()


def preprocess(img):
    face = DeepFace.extract_faces(img_path=img, grayscale=True)[0]['face']
    
    return face

def extract_vector(face):
    return model(face)

def avg_cosine_distance(v1, vectors):
    cosine_distances = [cosine(v1, v2) for v2 in vectors]
    avg_cosine_distance = np.mean(cosine_distances)
    return avg_cosine_distance

def farthest_vector(vectors):
    max_avg_cosine_distance = -np.inf 
    farthest_vec = None

    for i, v1 in enumerate(vectors):
        other_vectors = [v for j, v in enumerate(vectors) if j != i]
        curr_avg_cosine_distance = avg_cosine_distance(v1, other_vectors)
        
        if curr_avg_cosine_distance > max_avg_cosine_distance:
            max_avg_cosine_distance = curr_avg_cosine_distance
            farthest_vec = v1

    return farthest_vec

def analysis(img_list):
    vecs = []
    for idx, img in enumerate(img_list):
        vecs.append(extract_vector(preprocess(img)))
    return farthest_vector(vecs)
    
