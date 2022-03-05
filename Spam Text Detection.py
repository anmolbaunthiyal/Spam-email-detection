#!/usr/bin/env python
# coding: utf-8

# Importing Dependencies

# In[2]:



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Importing Dataset

# In[3]:


raw_data = pd.read_csv('E:\CSV_Datasets\spam_ham_dataset.csv')


# Replace the null values with null-string.

# In[4]:


mail_data = raw_data.where((pd.notnull(raw_data)),'')


# In[5]:


mail_data.head()


# In[6]:


# ROWS and COLUMNS of the dataset
mail_data.shape


# Label Encoding don't need in this case

# In[7]:


X = mail_data['text']
Y = mail_data['label_num']


# In[8]:


print(X)


# In[9]:


print(Y)


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=3)


#   Feature Extraction

# In[11]:


feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# Training Model
# Logistic Regression

# In[12]:


model = LogisticRegression()


# In[13]:


model.fit(X_train_features, Y_train)


#  Evaluating the trained Model

# In[14]:


# prediction on the trained data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)


# In[15]:


print("Accuracy on training data ",accuracy_on_training_data)


# In[16]:


#prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)


# In[17]:


print("Accuracy on test data : ",accuracy_on_test_data)


# BUILDING PREDICTIVE SYSTEM

# In[22]:


input_mail = [input()]
input_data_features = feature_extraction.transform(input_mail)
#making prediction
prediction = model.predict(input_data_features)
# print(prediction)
if prediction ==1 :
    print("SPAM DETECTED")
else :
    print("NOT SPAM")
    


# In[ ]:




