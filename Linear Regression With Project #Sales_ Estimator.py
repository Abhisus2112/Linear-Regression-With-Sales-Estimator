#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


# Simple Linear Regression

# In[2]:


data = pd.read_excel('F:/Python/Advertisement/Advertisement.xlsx')


# In[3]:


data


# In[4]:


#initializing the veriable
x=data['TV'].values.reshape(-1,1)
y=data['sales'].values.reshape(-1,1)


# In[5]:


x


# In[6]:


y


# In[7]:


#Ploting a graph to see the points
plt.figure(figsize=(16,8))
plt.scatter(x,y,c='black')
plt.xlabel('Money spenda on TV adds ($)')
plt.ylabel('Sales ($)')
plt.show()


# In[8]:


#Splitting our dataset to Traning and Testing Dataset

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=42)


# In[9]:


len(x_train)


# In[10]:


len(x_test)


# In[12]:


#fitting linear regression to the training set

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)


# In[13]:


#Predicting the Test set Result
y_pred = reg.predict(x_test)
plt.figure(figsize = (16,8))
plt.scatter(x,y, c='black')
plt.plot(x_test,y_pred, c='blue', linewidth=2)

plt.xlabel('Money spent for TV adds ($)')
plt.ylabel('Sales ($)')
plt.show()


# In[14]:


#calculate the coefficients

reg.coef_


# In[15]:


#Calculating the intercept

reg.intercept_


# In[16]:


#Calculating the R squared value

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[17]:


Output = reg.predict([[230.1]])
Output


# Multiple Linear Regression

# In[18]:


x = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)


# In[19]:


##Splitting our database to Training and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)


# In[20]:


#Spliting Linear regression to the Training set
from sklearn.linear_model import LinearRegression
multi_reg = LinearRegression()
multi_reg.fit(x_train, y_train)


# In[21]:


y_pred1 = multi_reg.predict(x_test)


# In[34]:


y_pred1


# In[22]:


multi_reg.coef_


# In[23]:


multi_reg.intercept_


# In[37]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred1)


# In[26]:


# taking the input from user

print("Enter the amount you will invest on:")
tv = float(input("TV : "))
radio = float(input("Radio : "))
newspaper = float(input("Newspaper : "))

#Predicting the sales with respect to inputs
output = multi_reg.predict([[tv,radio,newspaper]])
print("you will get Rs{:.2f} sales by advertising Rs{} on TV, Rs{} on Radio, Rs{} on Newspaper,"     .format(output[0][0] if output else "not predictable", tv,radio,newspaper))


# In[ ]:




