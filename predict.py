from classify import *
from keras.preprocessing import image

model = keras.models.load_model('RGB_Model.h5')

#Add the file to the project directory and rename the file as 'test.png'

prediction_image = image.load_img("test.png",target_size=(64, 64) ) 
prediction_image = image.img_to_array(prediction_image)
prediction_image = np.expand_dims(prediction_image,axis=0) 
answer = model.predict(prediction_image)

if(answer[0][0]==1):
    print("Dominant Color:-Blue")
elif(answer[0][1]==1):
    print("Dominant Color:-Green")
else:
    print("Dominant Color:-Red")
