# Imports
import random
import cv2
import os
import sys
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from tensorflow.python import keras
import pickle

import tensorflow.keras.models
from tensorflow.keras.models import model_from_json

import time, datetime

def create_cluster(folder_path,n_clusters,max_examples,output_folder):
    paths = os.listdir(folder_path)
    if max_examples == None:
        max_examples = len(paths)
    else:
        if max_examples > len(paths):
            max_examples = len(paths)
        else:
            max_examples = max_examples
    n_clusters = n_clusters
    folder_path = folder_path
    random.shuffle(paths)
    image_paths = paths[:max_examples]
    
    print(os.path.isdir(output_folder))
    
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
  
        
    os.makedirs(output_folder)
        
    
    print("\n output folders created.")
    
    for i in range(n_clusters):
        os.makedirs(output_folder + "\\cluster" + str(i))
    print("\n Object of class \"image_clustering\" has been initialized.")
    


def load_images(paths,folder_path):
    images_for_model=[]
    for i in paths:
        image_path = folder_path + '/' + str(i)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224,224))  # Resize it to 224 x 224
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert it from BGR to RGB so we can plot them later (because openCV reads images as BGR)
        images_for_model.append(image) # Now we add it to our array

    images_arr = np.array(images_for_model, dtype=np.float32)

    images_arr /= 255 # Normalise the images
    print(images_arr.shape)
    
    return images_arr

def covnet_transform(covnet_model, raw_images):

    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)
    
    return flat

def get_image_vectors(input_images, modelfile_json, model_fileh5):
    json_file = open(modelfile_json,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights(model_fileh5)
    
    densenet_output = covnet_transform(loaded_model, input_images)
    print("densenet flattened output has {} features".format(densenet_output.shape[1]))
    
    return densenet_output

def clustering(pkl_filename,input_images,dense_output,output_folder):
    with open(pkl_filename, 'rb') as file:
        kmeansmodel = pickle.load(file)
    predictions = kmeansmodel.predict(dense_output)
    for i in range(len(input_images)):
        path_1 = folder_path+"\\"+ paths[i]
        path_2 = output_folder + "\cluster"+ str(predictions[i])
        shutil.copy2(path_1, path_2)
    print("\n Clustering complete! \n\n Clusters and the respective images are stored in the \"output_clusters_densenet\" folder.")

def generate_plots_cluster(folder_path,output_folder, max_images, pkl_filename, modelfile_json, model_fileh5, n_clusters=10):
    paths = os.listdir(folder_path)
    print(len(paths))
    startime = time.time()       
    create_cluster(folder_path, n_clusters, max_images,output_folder)
    image_arr = load_images(paths)
    print ('laoding images took: ', time.time()-startime)
    startime = time.time()
    densenet_output = get_image_vectors(image_arr, modelfile_json, model_fileh5)
    print ('getting image vector from denset model took: ', time.time()-startime)
    startime = time.time()
    clustering(pkl_filename, image_arr, densenet_output, output_folder)
    print ('creating clusters using kmeans took: ', time.time()-startime)

def clusterassignment(pkl_filename,dense_output):
    with open(pkl_filename, 'rb') as file:
        print(pkl_filename)
        kmeansmodel = pickle.load(file)
    predictions = kmeansmodel.predict(dense_output)
    
    return predictions
    
    
    
def predicting_cluster(folder_path, pkl_filename, modelfile_json, model_fileh5):
    paths = os.listdir(folder_path)
    print(len(paths))      
    image_arr = load_images(paths,folder_path)
    startime = time.time()
    print ('laoding images took: ', time.time()-startime)
    densenet_output = get_image_vectors(image_arr, modelfile_json, model_fileh5)
    print ('getting image vector from denset model took: ', time.time()-startime)
    startime = time.time()
    result = clusterassignment(pkl_filename,densenet_output)
    print ('creating clusters using kmeans took: ', time.time()-startime)
    print(result)

    return result
                                                                                                                                                                                                                                                                                                    
if __name__ == "__main__":

    res = predicting_cluster('static/Obspy_Plots/diff_plots/predict','model_uploads/kmeans_model.pkl','model_uploads/densenetmodel.json','model_uploads/densenetmodel.h5')
    print(res)
    
    
    
    
