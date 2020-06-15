#Approach 1
#Created a dataset by webscarping images from shutterstock
#Drawbacks : Images not completely one color
#            Image were corrupted 


import requests 
from bs4 import BeautifulSoup 

headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'}

URL = "https://www.shutterstock.com/search/light+green+background+texture?coupon_code=pick10free&gclid=CjwKCAjw4pT1BRBUEiwAm5QuR083xYZ00tg8yAftX1KiDUZFsLVgOJ6IR5rdCReLcHR60w-uvUkC2xoC4rMQAvD_BwE&gclsrc=aw.ds"
req = requests.get(URL, headers = headers)

soup = BeautifulSoup(req.content, 'html5lib') 

img_src = []
img_div = list(soup.findAll('div', attrs = {'class' : 'z_h_f'}))

#extract href tags and then join href to get the link of image
for row in img_div:
    site = 'https://image.shutterstock.com'
    extra_text = '260nw-'
    ext = '.jpg'
    img_ref = row.a['href']
    ele = row.a['href'].split('-')
    num = ele[-1]
    img_str = img_ref.replace(num, '')
    img_src.append(site + img_str + extra_text + num + ext)

#to fill the incorrect images location
num_arr = []
count = 196
for i in range(len(img_src)):
    if i < len(num_arr):
        img_data = requests.get(img_src[i]).content
        with open('green/' + str(num_arr[i]) + '.jpg', 'wb') as handler:
            handler.write(img_data)
    else:
        img_data = requests.get(img_src[i]).content
        with open('green/' + str(count) + '.jpg', 'wb') as handler:
            handler.write(img_data)
        count = count + 1
    
    
