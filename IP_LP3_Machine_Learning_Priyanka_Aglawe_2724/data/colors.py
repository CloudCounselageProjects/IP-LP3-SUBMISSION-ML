import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def strim_string(string):
    
    string = string.replace(' ', '')
    string = string.replace('\\n', '')
    string = string.replace('\\t', '')

    return string

def create_color_image(height,
                       width,
                       red_min,
                       red_max,
                       green_min,
                       green_max,
                       blue_min,
                       blue_max,
                       num_images,
                       name,
                       path = None):
    
    """
    We provide images in BGR for OPENCV
    """
    
    img = np.array([0 for i in range(height*width*3)]).reshape(height,width,3).astype(np.uint8)
    
    img[:,:,2] = red_min # Red
    img[:,:,1] = green_min # Green
    img[:,:,0] = blue_min # Blue
    
    i = (red_max - red_min) // num_images
    j = (green_max - green_min) // num_images
    k = (blue_max - blue_min) // num_images
    
    index = 0
        
    for _ in range(0, num_images, 1):
        
        img[:,:,2] += i # Red
        img[:,:,1] += j # Green
        img[:,:,0] += k # Blue
        
        save_name = '{}-{}.jpg'.format(name, index)   
        save_path = save_name
        
        if path: 
            save_path = path + '/' + save_name

        while os.path.exists(save_path):
            index += 1
            save_name = '{}-{}.jpg'.format(name, index)   
            save_path = save_name
            
            if path:
                save_path = path + '/' + save_name
                
        cv2.imwrite(save_path, img)
        index += 1

def main():
    
    colors = pd.read_csv('colors.csv', header = None)
    
    for line in range(colors.shape[0]):
    
        lst = colors.iloc[line,:]
        lst[6] = strim_string(lst[6])
        
        create_color_image(64, 64,
                           int(lst[0]), int(lst[1]),
                           int(lst[2]), int(lst[3]),
                           int(lst[4]), int(lst[5]),
                           10,
                           lst[6])
        
if __name__ == 'main':
    
    main()