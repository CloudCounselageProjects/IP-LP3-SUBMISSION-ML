# Color Classifier

Color Classifier is an assignment of Cloud Counselage.
The assignment provided hands-on experience with tensorflow

## My approach
Initially I webscraped images from shutterstock, however while webscraping some images were corrupt and caused a bad prediction due to these images 

**To run the webscraping code**
```
python webscrape.py
```

I created images using cv2 where I create random colors 

**Adjust the parameters in the code  & execute image_maker**
 ```
 python image_maker.py
 ```

Then I trained a neural network model using tensorflow 2.0 and saved the model

**My method explained in the notebook ColorClassifer.ipynb**
 
 I have trained using 600 images and validated on 150 images
 
 Finally I wrote a script to predict Color of images in test dataset
 
 **Store the images in testing folder and execute image_predict**
 ```
 python image_predict.py
 ```
 


