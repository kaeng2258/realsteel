import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
    if os.path.exists('landmarks.npy') and os.path.exists('labels.npy'):
        landmark_list = np.load('landmarks.npy', allow_pickle=True).tolist()
        label_list = np.load('labels.npy', allow_pickle=True).tolist()
    else:
        landmark_list = []
        label_list = []
    return landmark_list, label_list

def save_data(landmark_list, label_list):
    np.save('landmarks.npy', np.array(landmark_list, dtype=np.float32))
    np.save('labels.npy', np.array(label_list))

def preprocess_data():
    landmarks = np.load('landmarks.npy')
    labels = np.load('labels.npy')
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return landmarks, labels, le
