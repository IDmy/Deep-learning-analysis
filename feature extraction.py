# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:52:58 2017

@author: DIhnatov
"""

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.preprocessing import image
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd

def feature_extractor(images_dir="PATH", model=""):
    list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
    labels = []
    features = []
    if model=="VGG16":
        target_size=(224, 224)
        model = VGG16(include_top=False, weights='imagenet', pooling='max')
    elif model=="ResNet50":
        target_size=(224, 224)        
        model = ResNet50(include_top=False, weights='imagenet', pooling='max')
    elif model=="Xception":
        target_size=(229, 229)
        model = Xception(include_top=False, weights='imagenet', pooling='max')
    elif model=="InceptionV3":
        target_size=(229, 229)
        model = InceptionV3(include_top=False, weights='imagenet', pooling='max')        
    else:
        target_size=(229, 229)
        model = InceptionV3(weights='imagenet', include_top=False, pooling='max')
    print ("Extracting features...")    
    for i in tqdm(list_images):
        img = image.load_img(i, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        image_id = i.split('/')[-1].split('.')[0]
        labels.append(image_id)
        features.append(feature)
    return labels, features
    
labels, features = feature_extractor('Path_to _the_folder_with_images', "InceptionV3")

# Save extracted features and labels separately in pickle format
pickle.dump(features, open('PATH'+'image_features', 'wb'))
pickle.dump(labels, open('PATH'+'image_labels', 'wb'))

# Save extracted features and labels together in csv
feature_arr = np.array(features)
feature_arr = np.reshape(feature_arr, (feature_arr.shape[0], feature_arr.shape[2])) 
feature_df = pd.DataFrame(feature_arr)
feature_df['label'] = labels
feature_df.to_csv('PATH'+'image_features_label.csv', index=False)
print ("Saved!")
