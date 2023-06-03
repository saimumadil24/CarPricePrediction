#!/usr/bin/env python
# coding: utf-8

# In[55]:


#Importing the libraries
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[56]:


#Importing the dataset and see the top five
data=pd.read_csv(r'C:\Users\Saimum Adil Khan\OneDrive\Desktop\Flask\Car Price Prediction\carprices.csv')
data.head()


# In[57]:


#Get the shape
data.shape


# In[58]:


#Checking the null values exist or not
data.isnull().sum()


# In[59]:


#taking the nuerical columns only
n_data=data.select_dtypes(include='number')
n_data

#boxplot of the numeric data
n_data.boxplot()


# In[63]:


#Checking the outlier
data[data['Sell Price']>50000]


# In[64]:


#Removing outlier
data=data[data['Sell Price']<50000]
data

#Adding dummy variable instead of Car Model
data=pd.get_dummies(data,columns=['Car Model'],drop_first=True)
data


# In[70]:


#Importing the Regression library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[71]:


#Taking the values of X and Y
X=data.drop(['Sell Price'],axis=1)
y=data['Sell Price']


# In[72]:


#dividing data into test and train data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[73]:


#LinearRegression
lr=LinearRegression()


# In[74]:


#Training the data with train data
lr.fit(X_train,y_train)


# In[75]:


#Making prediction with test data
lr.predict(X_test)


# In[76]:


#Checking accuracy score
lr.score(X_test,y_test)


# In[77]:


#Creating new dataframe
input_to_pred=pd.DataFrame(columns=X_train.columns)


# In[78]:


#Getting input  of new data of the dataframe
input_to_pred.loc[0]=[45000,8,0,0,0]
input_to_pred


# In[79]:


#Taking the prediction of price with the dataframe
price=lr.predict(input_to_pred)
price


# In[80]:


#Now adding the prediction of price in dataframe
final=input_to_pred
final[['Price']]=price.round(2)
final


# In[81]:


import pickle as pk


# In[83]:


with open('CarPrice.pkl','wb') as f:
    pk.dump(lr,f)

