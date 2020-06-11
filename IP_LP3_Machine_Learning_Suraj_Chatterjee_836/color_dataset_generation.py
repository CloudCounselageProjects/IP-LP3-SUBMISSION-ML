from colorsys import hsv_to_rgb
from PIL import Image

# Make some RGB values. 
# Cycle through hue vertically & saturation horizontally


count=10
x=0
y=0
z=0

for i in range(100):
  for j in range(100):
    colors = []
    for hue in range(100):
        for sat in range(100):
            # Convert color from HSV to RGB
            sat=i
            hue=j
            rgb = hsv_to_rgb(hue/100, sat/100, 1)
            rgb = [int(0.5 + 255*u) for u in rgb]
            
            # print(rgb)
            colors.extend(rgb)
            
    if(rgb[0]==rgb[1] and rgb[0]==rgb[2]):
      continue
    if(rgb[0]>rgb[1] and rgb[0]>rgb[2]):
        x+=1
        temp=str(x)
        colors = bytes(colors)
        img = Image.frombytes('RGB', (100, 100), colors)
        # img.show()
        if(x%3==0):
            t = 'D:/intern/CC/data_color/validate/red/'+ temp + '.png'
        else:
            t = 'D:/intern/CC/data_color/train/red/'+ temp + '.png'
        img.save(t)
        
    elif(rgb[1]>rgb[0] and rgb[1]>rgb[2]):
        y+=1
        temp=str(y)
        colors = bytes(colors)
        img = Image.frombytes('RGB', (100, 100), colors)
        # img.show()
        if(y%3==0):
            t = 'D:/intern/CC/data_color/validate/green/'+ temp + '.png'
        else:
            t = 'D:/intern/CC/data_color/train/green/'+ temp + '.png'
        img.save(t)
    else:
        z+=1
        temp=str(z)
        colors = bytes(colors)
        img = Image.frombytes('RGB', (100, 100), colors)
        # img.show()
        if(z%3==0):
            t = 'D:/intern/CC/data_color/validate/blue/'+ temp + '.png'
        else:
            t = 'D:/intern/CC/data_color/train/blue/'+ temp + '.png'
        img.save(t)

    # # Convert list to bytes
    # colors = bytes(colors)
    # img = Image.frombytes('RGB', (100, 360), colors)
    # img.show()
    # y = x + '.png'
    # img.save(y)