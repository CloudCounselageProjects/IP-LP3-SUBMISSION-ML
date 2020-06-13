


import pandas as pd
import math
from PIL import Image




names = ['Red','Green','Blue','output']

data=pd.read_csv(r"dataset.csv",names=names)

data1=data.values



def KNN(pred,data=data1,k=6):
    distances=list()
    votes=list()
    ans={}
    if k>len(data):
        return 'k can not be less than samples'
    for feature in data:
        euclidean_distance = math.sqrt((feature[0]-pred[0])**2 + (feature[1]-pred[1])**2  
                             + (feature[2]-pred[2])**2  )
        distances.append([euclidean_distance,feature[3]])
    distances.sort()   
    votes = [i[1] for i in distances[:k]]
    for i in votes:
        if i not in ans:
            ans[i]=0
        ans[i]+=1
    max=0
    for x, y in ans.items():
        if y > max:
            max=y
            key=x
    return key



# parameter is image path along name and extention,prints one of 'r', 'g', 'b'
def getinput(Image_path):
    im = Image.open(Image_path)
    im = im.convert('RGB')
    px=im.load()
    pix=px[0,0]
    op = KNN([pix[0],pix[1],pix[2]])
    print(op)
