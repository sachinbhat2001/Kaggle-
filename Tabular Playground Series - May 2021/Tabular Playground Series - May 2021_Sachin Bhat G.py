#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import library


# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt # plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#IMPORTING DATASETS
train_df='D:/kaggle/tabular-playground-series-may-2021/train.csv'
train=pd.read_csv(train_df)
test_df='D:/kaggle/tabular-playground-series-may-2021/test.csv'
test=pd.read_csv(test_df)


# In[3]:


#to check if any values are missing in training dataset
train.isnull().any()


# In[4]:


train.describe()


# In[5]:


train


# In[6]:


#label encoder- convert labels into numeric form

X = pd.DataFrame(train.drop(["target","id"], axis = 1))
lencoder = LabelEncoder()
y = pd.DataFrame(lencoder.fit_transform(train['target']), columns=['target'])


# In[7]:


y.shape
y


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape
X_train


# In[9]:


#COPY OF TEST DATASET WITHOUT 'id' COLUMN

test_copy = pd.DataFrame(test.drop("id", axis = 1)) 
test_copy.shape
test_copy.describe()


# In[10]:


#pip install xgboost


# In[11]:


#USING Extreme Gradient Boosting (xgb)


# In[12]:


model = xgb.XGBClassifier(objective='multi:softprob',use_label_encoder=False)
model.fit(X_train, y_train)


# In[13]:


y_pred=model.predict_proba(X_test)
y_pred
#y_pred=model.predict_proba(X_test)


# In[14]:


pred2 = model.predict_proba(test_copy)
pred2


# In[15]:


sub = pd.read_csv("sample_submission.csv")
predictions = pd.DataFrame(pred2, columns = ["Class_1", "Class_2", "Class_3", "Class_4"])
predictions['id'] = sub['id']
predictions


# In[16]:


fsize = (5,5)
predictions.drop(['id'],axis=1).hist(figsize=fsize)
predictions.to_csv("predictions.csv", index = False)


# In[ ]:





# In[ ]:




