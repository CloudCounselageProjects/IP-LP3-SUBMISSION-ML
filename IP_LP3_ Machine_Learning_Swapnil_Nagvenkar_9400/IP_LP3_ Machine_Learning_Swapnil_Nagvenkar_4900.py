#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("C:/Users/swapnil nagvenkar/Documents/Amazon - Movies and TV Ratings.csv")
df.head()


# In[3]:


new_df=df.fillna(' ')
new_df


# In[4]:


new_df.columns


# In[5]:


col=[]
for c in df.columns:
    col.append(c)
col
new_df.drop(columns=['Movie2', 'Movie3','Movie4', 'Movie5','Movie6', 'Movie7', 'Movie8', 'Movie9', 'Movie10', 'Movie11',
 'Movie12', 'Movie13', 'Movie14', 'Movie15', 'Movie16','Movie17', 'Movie18', 'Movie19', 'Movie20', 'Movie21', 'Movie22', 'Movie23', 'Movie24',
 'Movie25', 'Movie26', 'Movie27','Movie28', 'Movie29', 'Movie30', 'Movie31', 'Movie32', 'Movie33', 'Movie34', 'Movie35',
 'Movie36','Movie37', 'Movie38', 'Movie39', 'Movie40', 'Movie41', 'Movie42', 'Movie43', 'Movie44', 'Movie45',
 'Movie46', 'Movie47', 'Movie48', 'Movie49', 'Movie50', 'Movie51', 'Movie52', 'Movie53', 'Movie54', 'Movie55', 'Movie56',
 'Movie57', 'Movie58','Movie59','Movie60', 'Movie61', 'Movie62', 'Movie63', 'Movie64', 'Movie65', 'Movie66', 'Movie67', 'Movie68',
 'Movie69', 'Movie70', 'Movie71', 'Movie72', 'Movie73', 'Movie74', 'Movie75', 'Movie76', 'Movie77', 'Movie78', 'Movie79',
 'Movie80', 'Movie81', 'Movie82', 'Movie83','Movie84', 'Movie85', 'Movie86', 'Movie87','Movie88', 'Movie89', 'Movie90', 'Movie91',
 'Movie92', 'Movie93', 'Movie94', 'Movie95', 'Movie96', 'Movie97', 'Movie98', 'Movie99', 'Movie100', 'Movie101', 'Movie102', 'Movie103',
 'Movie104', 'Movie105', 'Movie106', 'Movie107', 'Movie108', 'Movie109', 'Movie110', 'Movie111', 'Movie112', 'Movie113',
 'Movie114', 'Movie115', 'Movie116', 'Movie117', 'Movie118', 'Movie119', 'Movie120', 'Movie121', 'Movie122', 'Movie123',
 'Movie124', 'Movie125', 'Movie126', 'Movie127', 'Movie128', 'Movie129', 'Movie130', 'Movie131', 'Movie132', 'Movie133', 'Movie134',
 'Movie135', 'Movie136','Movie137', 'Movie138', 'Movie139', 'Movie140', 'Movie141', 'Movie142', 'Movie143', 'Movie144', 'Movie145',
 'Movie146', 'Movie147', 'Movie148', 'Movie149', 'Movie150', 'Movie151', 'Movie152', 'Movie153', 'Movie154', 'Movie155', 'Movie156',
 'Movie157', 'Movie158', 'Movie159', 'Movie160', 'Movie161', 'Movie162', 'Movie163', 'Movie164', 'Movie165', 'Movie166', 'Movie167', 'Movie168', 'Movie169',
 'Movie170', 'Movie171', 'Movie172', 'Movie173', 'Movie174', 'Movie175', 'Movie176', 'Movie177', 'Movie178', 'Movie179', 'Movie180',
 'Movie181', 'Movie182', 'Movie183', 'Movie184', 'Movie185', 'Movie186', 'Movie187', 'Movie188', 'Movie189', 'Movie190', 'Movie191',
 'Movie192', 'Movie193', 'Movie194', 'Movie195', 'Movie196', 'Movie197', 'Movie198', 'Movie199', 'Movie200', 'Movie201', 'Movie202',
 'Movie203', 'Movie204', 'Movie205', 'Movie206'] ,axis=1)


# In[6]:


plt.scatter(df.user_id,df.Movie1,marker='*',color='red')


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(df[['user_id']],df.Movie1,test_size=0.1)


# In[8]:


x_train


# In[9]:


y_test


# In[10]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()


# In[11]:


model= LogisticRegression()


# In[ ]:





# In[ ]:




