import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump, load
import pickle
import warnings

def read_img(directory = 'data'):
 
    images = glob.glob(directory + '/*.jpg')
    X, y = [], []
    
    for image in images:
        
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = image.split('\\')[-1]
        label = label.split('.')[0]
        label = label.split('-')[0]
    
        X.append(img)
        y.append(label)
    
    X = np.array(X).reshape(-1, 64, 64, 3).astype(np.float32)
    y = np.array(y).reshape(-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size = 0.1, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def create_one_hot_encoder(y, 
                           enc1_file = 'label_encoder.pkl',
                           enc2_file = 'hot_encoder.pkl'):
    
    if os.path.exists(enc1_file) and os.path.exists(enc2_file):
        print("The pickle files already exists")
        return False
    
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(y)
    encoded = encoded.reshape(len(encoded), 1)
    
    enc = OneHotEncoder(sparse=False)
    enc.fit(encoded)
    
    with open(enc1_file, 'wb') as write:
        pickle.dump(label_encoder, write)
    
    with open(enc2_file, 'wb') as write:
        pickle.dump(enc, write)
        
    print("Model Saved Sucessfully")
    return True
    
def load_encoder_output(y, 
                        enc1_file = 'label_encoder.pkl',
                        enc2_file = 'hot_encoder.pkl'):
    
    if not os.path.exists(enc1_file) and not os.path.exists(enc2_file):
        print("These Files does not exists")
        return False
    
    with open(enc1_file, 'rb') as read:
        encoder = pickle.load(read)
        
    y_output = encoder.transform(y)
    y_output = y_output.reshape(len(y_output), 1)
    
    with open(enc2_file, 'rb') as read:
        encoder = pickle.load(read)
        
    y_output = encoder.transform(y_output)
    
    return y_output

def one_hot_decoder(y,
                    enc1_file = 'label_encoder.pkl',
                    enc2_file = 'hot_encoder.pkl'):
    
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    
    if not os.path.exists(enc1_file) and not os.path.exists(enc2_file):
        print("These Files does not exists")
        return False
    
    with open(enc1_file, 'rb') as read:
        encoder = pickle.load(read)

    y_output = encoder.inverse_transform(y)
    
    return y_output