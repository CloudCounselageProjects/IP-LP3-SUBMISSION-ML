# LP3_internship-COLOR RECONGNITION MODEL

The repository is for color recongnition model using tensorflow and SVM <br />
There are folders named 'red','green','blue' which contains the images of training dataset.<br />

1.There is a folder named "using_tensorflow" which has a color recongnition using tensorflow DNN classifier.<br />
  The first python file "create_dataset" creates a csv file 'p3.csv' with the pixel values for each training image.Here in the class column 0=red      color 1=green 2=blue   <br />
  The second program is the code for the  model to recongnize the color.<br />
  **Note:This requires the tensorflow 1.14.0 version** <br />
  To install tensorflow 1.14.0  using conda <br />
    
    conda uninstall tensorflow
    conda install -c conda-forge tensorflow=1.14.0
    
 
  Below is the screenshot for the test image given as input
  ![output1](https://user-images.githubusercontent.com/62999002/83961114-b9e7b480-a8ad-11ea-84be-1a449271a0d0.jpg)

  
2.The other folder "using_SVM" has a color recongnition model  using sklearn library and SVM .<br />
  The first python file "create_dataset" creates a csv file 'p3.csv' with the pixel values for each training image.Here in the class column 0=red      color 1=green 2=blue   <br />
   
  <br />
  Below is the screenshot for the test image given as input
  
  
  ![output2](https://user-images.githubusercontent.com/62999002/83961223-b4d73500-a8ae-11ea-98e6-053e86e1d292.jpg)
 <br />
 The image actual_ip is the image given as the test_image according to the LP3 statement
  <br />
   
