# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:16:52 2020

@author: varsh
"""

from PIL import Image
import random
count={'red':0,'green':0,'blue':0}
for i in range(100):
    dict={'red':random.randint(0,255),'green':random.randint(0,255),'blue':random.randint(0,255)}
    img=Image.new('RGB',(64,64),color=(dict['red'],dict['green'],dict['blue']))
    colour=max(dict,key=dict.get)
    count[colour]+=1
    m=count[colour]
    #print(colour)
    #print(count)
   
    #img.show()
    img.save('C:\\Users\\varsh\Desktop\color classifier\dataset\\validation\\'+str(colour)+'\\'+str(m)+'.jpg')
    
    

    