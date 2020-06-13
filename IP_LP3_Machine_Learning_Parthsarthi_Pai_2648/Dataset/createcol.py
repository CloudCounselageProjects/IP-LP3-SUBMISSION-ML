import os
from PIL import Image
count=0

for r in range (0,255,10):
    for b in range(0,255,20):
        for g in range(0,255,20):
            if r>b and r>g:
                img= Image.new('RGB' , (28,28), (r,g,b))
                count+=1
                dire=r"D:\images\red"
                full_path=os.path.join(dire,str(count))
                img.save(full_path+'.png', 'JPEG')
            elif g>b and r<g:
                img=Image.new('RGB', (28,28), (r,g,b))
                count+=1
                dire=r"D:\images\green"
                full_path=os.path.join(dire,str(count))
                img.save(full_path+'.png', 'JPEG')
            elif r<b and b>g:
                img=Image.new('RGB',(28,28), (r,g,b))
                count+=1
                dire=r"D:\images\blue"
                full_path=os.path.join(dire,str(count))
                img.save(full_path+'.png', 'JPEG')

