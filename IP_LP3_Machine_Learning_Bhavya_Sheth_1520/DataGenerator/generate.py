from PIL import Image, ImageColor
import random
import os
import shutil

trainPath = '../Data/train'
testPath = '../Data/test'

def generateFolder():
    try:
        shutil.rmtree(testPath)
        shutil.rmtree(trainPath)
    
    except:
        pass

    if not os.path.exists(trainPath):
        os.makedirs(trainPath)

    if not os.path.exists(testPath):
        os.makedirs(testPath)

    # shutil.rmtree(testPath)
    

def getRgbColor():
    colors = {
    'blue' : [0,0,random.randint(150,255)],
    'red': [random.randint(150,255),0,0],
    'orange': [255,random.randint(50,255),0],
    'green': [0,random.randint(150,255),0],
    'violet': [random.randint(230,245),random.randint(120,140),random.randint(230,245)],
    'indigo': [random.randint(70,85) , 0, random.randint(120,140)],
    'yellow': [random.randint(200,256),random.randint(200,256),0]
    }

    color = random.choice(list(colors))
    return (color, colors[color])

def generateData(n, path):
    width = 28
    height = 28

    labels = open(path + "/labels.txt", "w")

    for i in range(n):
        color = getRgbColor()
        # print(color[0])

        im = Image.new("RGB", (width, height), tuple(color[1]))
        im.save(path + '/im' + str(i), format='png')

        labels.write(color[0] + "\n")

    labels.close()

generateFolder()
generateData(500, trainPath)
generateData(50, testPath)