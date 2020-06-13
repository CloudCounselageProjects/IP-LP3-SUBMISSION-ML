
#Import required Library
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_resource_variables()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


#Input image
temp_img=cv2.imread("redsmoke.png")
temp_img = cv2.resize(temp_img, (0, 0), fx = 0.5, fy = 0.5)
img=cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB)



#
num_points = 100
dimensions = 2

points =np.array(img)[:,:,:3]

#changing the image shape
pixels = np.float32(img.reshape(-1, 3))




#converting the given image data to tensors
def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(pixels, dtype=tf.float32), num_epochs=1)
#define the number of clusters
num_clusters = 3
#Using kmeans library
kmeans = tf.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False)



# train
num_iterations = 1
previous_centers = None
for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print ('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers





# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
counts=np.unique(cluster_indices, return_counts=True)[1]
palette=cluster_centers
#finding the dominant pixels
dominant = palette[np.argmax(counts)]
'''for i, point in enumerate(points):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]'''
#calculating the dominant color IN RGB format
minValue=dict()
red=[255,0,0]
green=[0,255,0]
blue=[0,0,255]
r=g=b=0
for i in range(3):
    r+=(abs(dominant[i]-red[i]))
    g+=(abs(dominant[i]-green[i]))
    b+=(abs(dominant[i]-blue[i]))
minValue["RED"]=r
minValue["GREEN"]=g
minValue["BLUE"]=b

minValue=sorted(minValue.items(),key=lambda x:x[1])


for elem in minValue:
    print(elem[0])

    break
